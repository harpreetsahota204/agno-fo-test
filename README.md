# Example Agno Agent (powered by Gemini) Plugin

![](assets/agno_agent_app.gif)

A FiftyOne plugin using an Agno agent that enables object detection using Google's Gemini vision-language model. This plugin provides a user-friendly operator interface to detect objects in images using natural language prompts.

## Installation

## Setup environment variable

Before using the plugin, you'll need to set up your Gemini API key:
   
1. Set the environment variable before running your Python script:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

2. **In Python (Interactive)**
   
   Set it during your interactive session using getpass:
   ```python
   import os
   from getpass import getpass

   os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API key: ")
   ```

3. **In Python (Direct)**
   
   Set it directly in your code (Note: not recommended for shared code):
   ```python
   import os
   
   os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
   ```
## Plugin installation

1.  Install the plugin directly from GitHub:
```bash
fiftyone plugins download https://github.com/harpreetsahota204/agno-fo-test
```

2. To install requirements:
```bash
fiftyone plugins requirements @harpreetsahota/agnoagent --install
```

## Usage

### Through FiftyOne App

```python
import fiftyone as fo

# Load your dataset
dataset = fo.load_dataset("your-dataset")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

Then use the "Ageno Agent" operator from the App interface.


### Programmatic Usage

```python
import fiftyone.operators as foo

# Get the operator
agno_agent = foo.get_operator("@harpreetsahota/agnoagent/agno_agent")

# Example system message
SYSTEM_MESSAGE = """Given an image of a GUI screenshot and a corresponding instruction, the task is to  output one bounding box for the relevant GUI element in the screenshot that correspond to the instruction and  associated with one of the following labels: text or icon. 
"""

# Method 1: Using field-based prompts
agno_agent(
    dataset,
    operation_mode="prompt_field",
    output_field="detections",      # where to store results
    system_message=SYSTEM_MESSAGE,
    prompt_field="instruction",     # field containing prompts
    verbose=True,                   # enable debugging output
)

# Method 2: Using a single text prompt for all images
agno_agent(
    dataset,
    operation_mode="text_prompt",
    output_field="detections",      # where to store results
    system_message=SYSTEM_MESSAGE,
    text_prompt="Find all buttons in this image",  # prompt applied to all images
    verbose=True,                   # enable debugging output
)
```
## Running Delegated Service

You can also choose to run this as a delegated operation. To do so you must first set the following environment variable (either in terminal via `export` or in Python) before you instantiate the operator:

```python
import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'
```

Then, in the terminal,  run `fiftyone delegated launch`. When running a delegated operation in a notebook you will need to use the `await` syntax:

```python
await agno_agent(
    dataset,
    operation_mode="text_prompt",
    output_field="detections",      # where to store results
    system_message=SYSTEM_MESSAGE,
    text_prompt="Find all buttons in this image",  # prompt applied to all images
    verbose=True,                   # enable debugging output
    delegate=True
)
```

## Configuration Options

### Operation Modes

1. **Field-based Prompts** (`prompt_field`):
   - Uses sample-specific prompts stored in a dataset field
   - Specify the field name containing prompts for each image

2. **Single Text Prompt** (`text_prompt`):
   - Uses the same prompt for all images
   - Provide one prompt that applies to all samples

### Parameters
- `operation_mode`: The operation mode you want to use
- `output_field`: Name of the field where detections will be stored
- `system_message`: Instructions for the Gemini model
- `verbose`: Enable detailed logging (default: False)
- `delegate`: Run on delegated service (default: False)
