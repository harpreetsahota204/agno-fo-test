import json
import re
import numpy as np
from typing import Union, Dict, Optional, List

# FiftyOne imports
import fiftyone as fo
from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Detection, Detections

# Agno imports
from agno.agent import Agent, RunResponse
from agno.media import Image
from agno.models.google import Gemini

class AgnoAgent(Model, SamplesMixin):
    """A model that uses Gemini to detect objects based on text instructions."""
    
    def __init__(self, model_id, system_message):
        """Initialize the Detection model."""
        
        # Initialize SamplesMixin
        self._fields = {}
        self.prompt_field = None
        
        # Store parameters
        self.model_id = model_id
        self.system_message = system_message
        
        # Initialize Agno agent with Gemini - use JSON output format
        self.agent = Agent(
            model=Gemini(id=model_id),
            system_message=system_message,
            structured_outputs=False,  # Set to False since we'll parse JSON manually
        )
        
        # Flag for debugging
        self.verbose = False
    
    @property
    def media_type(self):
        """The media type processed by the model."""
        return "image"
    
    def convert_bbox_format(self, bbox):
        """Convert bounding box coordinates."""
        y1, x1, y2, x2 = bbox
        
        # Convert directly from Gemini's 1000x1000 grid to relative coordinates
        rel_x = x1 / 1000
        rel_y = y1 / 1000
        rel_width = (x2 - x1) / 1000
        rel_height = (y2 - y1) / 1000
        
        return [rel_x, rel_y, rel_width, rel_height]
    
    def _extract_json_from_response(self, text):
        """Extract JSON from Gemini's response which may contain markdown code blocks."""
        # Pattern to find JSON inside markdown code blocks
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(code_block_pattern, text)
        
        if match:
            # Return the content inside the code block
            return match.group(1).strip()
        else:
            # If no code block is found, assume the entire text might be JSON
            return text.strip()
    
    def _process_single_sample(self, sample, prompt):
        """
        Process a single sample with the given prompt.
        Helper method used by both paths.
        """
        # Use sample's filepath
        if not hasattr(sample, 'filepath'):
            raise ValueError(f"Sample {sample.id} must have a valid filepath attribute")
            
        filepath = sample.filepath
        
        # Use the exact prompt provided by the user without modification
        # Run Gemini agent
        response = self.agent.run(
            prompt,  # Use the prompt exactly as provided
            images=[Image(filepath=filepath)]
        )
        
        # Debug output if verbose
        if self.verbose:
            print(f"Sample: {sample.id}")
            print(f"Prompt: {prompt}")
            print(f"Raw response: {response.content}")
        
        # Process the result
        detections = []
        
        # Extract JSON from response
        if hasattr(response, 'content') and isinstance(response.content, str):
            try:
                # Extract JSON text from possible markdown
                json_text = self._extract_json_from_response(response.content)
                
                # Parse the JSON
                detection_list = json.loads(json_text)
                
                # Process each detection
                if isinstance(detection_list, list):
                    for detection in detection_list:
                        try:
                            if "box_2d" in detection and "label" in detection:
                                # Convert bbox format
                                bbox = self.convert_bbox_format(detection["box_2d"])
                                
                                # Create Detection object
                                detection_obj = Detection(
                                    label=detection["label"],
                                    bounding_box=bbox
                                )
                                detections.append(detection_obj)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing detection: {e}")
                            continue
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Raw response: {response.content}")
        
        return Detections(detections=detections)
    
    def predict(self, image, sample):
        """Detect objects based on text instruction."""
        
        # The prompt_field should be set by the constructor or external code
        prompt_field = None
        
        # First, check if prompt_field is in kwargs
        if "prompt_field" in self._fields:
            prompt_field = self._fields["prompt_field"]
        
        # Use the instance variable as fallback
        if prompt_field is None and self.prompt_field is not None:
            prompt_field = self.prompt_field
        
        if prompt_field is None:
            error_msg = "prompt_field must be specified"
            raise ValueError(error_msg)
        
        # Check if sample has the required field
        try:
            prompt = sample[prompt_field]
        except Exception as e:
            error_msg = f"Error accessing field '{prompt_field}' in sample {sample.id}: {e}"
            raise ValueError(error_msg)
        
        if prompt is None:
            error_msg = f"Sample {sample.id} field '{prompt_field}' is None"
            raise ValueError(error_msg)
            
        # Process the sample using the helper method
        return self._process_single_sample(sample, prompt)
    
    def predict_all(self, images, samples):
        """Batch prediction with sample fields."""
        results = []
        
        for sample in samples:
            try:
                result = self.predict(None, sample=sample)
                results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"Error predicting sample {sample.id}: {e}")
                results.append(Detections())
        
        return results
    
    def run_agent(self, dataset, label_field, prompt_field=None, text_prompt=None):
        """
        Entry point method that handles both prompt_field and text_prompt approaches.
        
        Args:
            dataset: FiftyOne dataset to process
            label_field: Field name to store detection results
            prompt_field: Field name containing prompts for each sample
            text_prompt: Fixed text prompt to use for all samples
            
        Returns:
            The updated dataset
        """
        # Validate parameters
        if prompt_field is not None and text_prompt is not None:
            raise ValueError("Cannot specify both prompt_field and text_prompt. Choose one.")
        if prompt_field is None and text_prompt is None:
            raise ValueError("Either prompt_field or text_prompt must be specified.")
        
        if prompt_field is not None:
            # PATH 1: Using field-specific prompts
            if self.verbose:
                print(f"Using sample-specific prompts from field '{prompt_field}'")
            
            # Ensure the field exists in the dataset
            if not dataset.has_sample_field(prompt_field):
                raise ValueError(f"Field '{prompt_field}' does not exist in the dataset.")
            
            # Process each sample manually, reading the prompt from the field
            for sample in dataset:
                try:
                    # Get prompt from the field - access the dataset directly instead of the view
                    sample_from_dataset = dataset[sample.id]
                    if prompt_field not in sample_from_dataset:
                        if self.verbose:
                            print(f"Warning: Field '{prompt_field}' not in sample {sample.id}")
                        continue
                        
                    prompt = sample_from_dataset[prompt_field]
                    if prompt is None:
                        if self.verbose:
                            print(f"Warning: Sample {sample.id} field '{prompt_field}' is None")
                        continue
                    
                    # Process the sample with its specific prompt
                    detections = self._process_single_sample(sample, prompt)
                    
                    # Save results
                    sample[label_field] = detections
                    sample.save()
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing sample {sample.id}: {e}")
                    # Initialize empty detections
                    sample[label_field] = Detections()
                    sample.save()
            
        else:  # text_prompt is not None
            # PATH 2: Using a fixed text prompt
            if self.verbose:
                print(f"Using fixed text prompt: '{text_prompt}'")
            
            # Process each sample manually with the same prompt
            for sample in dataset:
                try:
                    # Process the sample with the fixed prompt
                    detections = self._process_single_sample(sample, text_prompt)
                    
                    # Save results
                    sample[label_field] = detections
                    sample.save()
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing sample {sample.id}: {e}")
                    # Initialize empty detections
                    sample[label_field] = Detections()
                    sample.save()
        
        return dataset