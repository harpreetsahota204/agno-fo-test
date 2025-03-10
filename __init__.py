import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import requests 

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

from .agnoagent import AgnoAgent

def _handle_calling(
        uri, 
        sample_collection, 
        operation_mode,
        output_field,
        system_message,
        text_prompt=None,
        prompt_field=None,
        verbose=False,
        delegate=False,
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        operation_mode=operation_mode,
        output_field=output_field,
        system_message=system_message,
        text_prompt=text_prompt,
        prompt_field=prompt_field,
        verbose=verbose,
        delegate=delegate,
        )
    return foo.execute_operator(uri, ctx, params=params)

# Define the operation modes
OPERATION_MODES = {
    "text_prompt": "Use a single text prompt for all images", 
    "prompt_field": "Use sample-specific prompts from a field",
}

class AgnoDetectionOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="agno_detection",
            label="Ageno Agent",
            description="Run a Gemini powered Agno Agent for object detection on your Dataset!",
            dynamic=True,
            icon="assets/agent-detective-svgrepo-com.svg",
            )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        mode_dropdown = types.Dropdown(label="How would you like to provide prompts?")
        
        for k, v in OPERATION_MODES.items():
            mode_dropdown.add_choice(k, label=v)

        inputs.enum(
            "operation_mode",
            values=mode_dropdown.values(),
            label="Prompt Mode",
            description="Select how you want to provide detection prompts",
            view=mode_dropdown,
            required=True
        )
        
        # Get the chosen operation mode
        chosen_mode = ctx.params.get("operation_mode")

        if chosen_mode == "text_prompt":
            inputs.str(
                "text_prompt",
                label="Detection Prompt",
                description="Prompt to use for all images (e.g., 'Find all cars in this image')",
                required=True,
            )
        elif chosen_mode == "prompt_field":
            inputs.str(
                "prompt_field",
                label="Prompt Field",
                description="Name of the field that contains detection prompts for each sample",
                required=True,
            )

        inputs.str(
            "system_message",
            label="System Message",
            description="System message that guides the model's behavior. Must instruct the model to return results in JSON format with 'box_2d' coordinates in a 1000x1000 grid and 'label' fields.",
            multiline=True,
            required=True,
        )

        inputs.str(
            "output_field",            
            required=True,
            label="Output Field",
            description="Name of the field to store the detection results"
        )
        
        inputs.bool(
            "verbose",
            default=False,
            required=False,
            label="Verbose mode",
            description="Enable detailed logging for debugging",
            view=types.CheckboxView(),
        )
        
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        operation_mode = ctx.params.get("operation_mode")
        system_message = ctx.params.get("system_message")
        output_field = ctx.params.get("output_field")
        verbose = ctx.params.get("verbose", False)
        
        # Create the AgnoAgent model
        model = AgnoAgent(
            model_id="gemini-2.0-flash",  # Fixed model
            system_message=system_message
        )
        
        # Set verbose mode
        model.verbose = verbose
        
        # Run the agent based on operation mode
        if operation_mode == "text_prompt":
            text_prompt = ctx.params.get("text_prompt")
            model.run_agent(
                dataset=view,
                label_field=output_field,
                text_prompt=text_prompt
            )
        elif operation_mode == "prompt_field":
            prompt_field = ctx.params.get("prompt_field")
            model.run_agent(
                dataset=view,
                label_field=output_field,
                prompt_field=prompt_field
            )
        
        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            operation_mode,
            output_field,
            system_message,
            text_prompt=None,
            prompt_field=None,
            verbose=False,
            delegate=False,
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            operation_mode,
            output_field,
            system_message,
            text_prompt,
            prompt_field,
            verbose,
            delegate,
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(AgnoDetectionOperator)