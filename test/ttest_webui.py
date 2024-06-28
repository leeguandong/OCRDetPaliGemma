"""PaliGemma demo gradio app."""

import functools
import logging
import gradio as gr
import PIL.Image
from utils import parse

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

INTRO_TEXT = """OCRPaliGemma demo\n\n
| [GitHub](https://github.com/leeguandong/OCRPaliGemma) 
| [CSDN blog post](https://huggingface.co/blog/paligemma) 
\n\n
**This is an experimental research model.** Make sure to add appropriate guardrails when using the model for applications.
"""
MODELS_INFO = {
    'paligemma-3b-mix-224': (
        'JAX/FLAX PaliGemma 3B weights, finetuned with 224x224 input images and 256 token input/output '
        'text sequences on a mixture of downstream academic datasets. The models are available in float32, '
        'bfloat16 and float16 format for research purposes only.'
    ),
    'paligemma-3b-mix-448': (
        'JAX/FLAX PaliGemma 3B weights, finetuned with 448x448 input images and 512 token input/output '
        'text sequences on a mixture of downstream academic datasets. The models are available in float32, '
        'bfloat16 and float16 format for research purposes only.'
    ),
}

make_image = lambda value, visible: gr.Image(
    value, label='Image', type='filepath', visible=visible)
make_annotated_image = functools.partial(gr.AnnotatedImage, label='Image')
make_highlighted_text = functools.partial(gr.HighlightedText, label='Output')

# https://coolors.co/4285f4-db4437-f4b400-0f9d58-e48ef1
COLORS = ['#4285f4', '#db4437', '#f4b400', '#0f9d58', '#e48ef1']

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)


def generate(sampler: str, image: PIL.Image, prompt: str):
    inputs = processor(prompt, image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=20)

    return processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
    # pass


def compute(image, prompt, model_name, sampler):
    """Runs model inference."""
    if image is None:
        raise gr.Error('Image required')

    logging.info('prompt="%s"', prompt)

    if isinstance(image, str):
        image = PIL.Image.open(image)

    if not model_name:
        raise gr.Error('Models not loaded yet')
    output = generate(sampler, image, prompt)
    logging.info('output="%s"', output)

    width, height = image.size
    objs = parse.extract_objs(output, width, height, unique_labels=True)
    labels = set(obj.get('name') for obj in objs if obj.get('name'))
    color_map = {l: COLORS[i % len(COLORS)] for i, l in enumerate(labels)}
    highlighted_text = [(obj['content'], obj.get('name')) for obj in objs]
    annotated_image = (
        image,
        [
            (
                obj['mask'] if obj.get('mask') is not None else obj['xyxy'],
                obj['name'] or '',
            )
            for obj in objs
            if 'mask' in obj or 'xyxy' in obj
        ],
    )
    has_annotations = bool(annotated_image[1])
    return (
        make_highlighted_text(
            highlighted_text, visible=True, color_map=color_map),
        make_image(image, visible=not has_annotations),
        make_annotated_image(
            annotated_image, visible=has_annotations, width=width, height=height,
            color_map=color_map),
    )


def warmup(model_name):
    image = PIL.Image.new('RGB', [1, 1])
    _ = compute(image, '', model_name, 'greedy')


def reset():
    return (
        '', make_highlighted_text('', visible=False),
        make_image(None, visible=True), make_annotated_image(None, visible=False),
    )


def create_app():
    """Creates demo UI."""

    make_model = lambda choices, visible=True: gr.Dropdown(
        value=(choices + [''])[0],
        choices=choices,
        label='Model',
        visible=visible,
    )
    make_prompt = lambda value, visible=True: gr.Textbox(
        value, label='Prompt', visible=visible)

    with gr.Blocks() as demo:
        ##### Main UI structure.

        gr.Markdown(INTRO_TEXT)
        with gr.Row():
            image = make_image(None, visible=True)  # input
            annotated_image = make_annotated_image(None, visible=False)  # output
            with gr.Column():
                with gr.Row():
                    prompt = make_prompt('', visible=True)
                model_info = gr.Markdown(label='Model Info')
                with gr.Row():
                    model = make_model([])
                    samplers = [
                        'greedy', 'nucleus(0.1)', 'nucleus(0.3)', 'temperature(0.5)']
                    sampler = gr.Dropdown(
                        value=samplers[0], choices=samplers, label='Decoding'
                    )
                with gr.Row():
                    run = gr.Button('Run', variant='primary')
                    clear = gr.Button('Clear')
                highlighted_text = make_highlighted_text('', visible=False)

        ##### UI logic.

        def update_ui(model, prompt):
            prompt = make_prompt(prompt, visible=True)
            model_info = f'Model `{model}` – {MODELS_INFO.get(model, "No info.")}'
            return [prompt, model_info]

        gr.on(
            [model.change],
            update_ui,
            [model, prompt],
            [prompt, model_info],
        )

        gr.on(
            [run.click, prompt.submit],
            compute,
            [image, prompt, model, sampler],
            [highlighted_text, image, annotated_image],
        )
        clear.click(
            reset, None, [prompt, highlighted_text, image, annotated_image]
        )

    return demo


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    create_app().queue().launch()
