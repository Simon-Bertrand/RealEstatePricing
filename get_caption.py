import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# we associate a model with its preprocessors to make it easier for inference.
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device
)
# uncomment to use base model
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="base_coco", is_eval=True, device=device
# )
vis_processors.keys()


def get_caption(image, device):

    raw_image = Image.fromarray(image)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # we'll first use beam search to generate a caption
    caption = model.generate({"image": image})

    # we'll then use nucleus sampling to get multiple captions
    # and more precise captions (you can uncomment the line below)

    # caption = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)

    return caption


def get_caption2(image, device):

    raw_image = Image.open(image).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # we'll first use beam search to generate a caption
    caption = model.generate({"image": image})

    # we'll then use nucleus sampling to get multiple captions
    # and more precise captions (you can uncomment the line below)

    # model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)

    # store the generated caption in a variable
    # caption = model.generated_captions[0]
    return caption

