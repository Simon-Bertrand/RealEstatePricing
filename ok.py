import torch, sys, torchvision
from PIL import Image
from dataloader.get import DataGetter
sys.path.append('./LAVIS/')
from lavis.models import load_model_and_preprocess

dg=DataGetter()
print(torch.cuda.is_available())

# setup device to use
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print(device)

# we associate a model with its preprocessors to make it easier for inference.
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="large_coco", is_eval=True, device=device
# )
# uncomment to use base model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=True, device=device
)
vis_processors.keys()

def get_caption(image,device):
    
    raw_image = Image.fromarray(image)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    
    # we'll first use beam search to generate a caption
    caption = model.generate({"image": image})
    
    # we'll then use nucleus sampling to get multiple captions
    # and more precise captions (you can uncomment the line below)
    
    # caption = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)
    
    # store the generated caption in a variable
    return caption


    
    
import asyncio
import time

async def get_caption_async(image):
    captions = get_caption(image, device)
    return captions

async def get_captions_batch_async(row, device):
    tasks = []
    for im in row.load():
        tasks.append(get_caption_async(im))
    captions = await asyncio.gather(*tasks)
    return captions



async def get_column_for_captions_async(ds, batch_size=50):
    captions = []
    all_captions=[]
    dj=ds['images']
    
    for i in range(len(dj)):
        batch_captions = await get_captions_batch_async(dj.iloc[i], device)
        all_captions.append(batch_captions)
    
    return all_captions





if __name__ == '__main__':
    df=DataGetter()
    df=df.getData('train')
    
    num_test = 100
    
    ds=df[:num_test]
    
    start = time.time()
    
    all_captions = asyncio.run(get_column_for_captions_async(ds, batch_size=50))
    ds['captions']=all_captions
    
    # time in minutes
    print((time.time() - start)/60)
    
    
    # print(ds)
    ds.to_csv('tab.csv')

