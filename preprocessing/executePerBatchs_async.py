
import torch, sys, torchvision
from PIL import Image
from dataloader.get import DataGetter
sys.path.append('./LAVIS/')
from lavis.models import load_model_and_preprocess

import asyncio
import time

df=DataGetter()
df=df.getData('train')
device = torch.device("cuda")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=True, device=device
)
vis_processors.keys()

def get_caption(image, model, vis_processors, device):
    return model.generate({
          "image":  vis_processors["eval"](Image.fromarray(image)).unsqueeze(0).to(device)
          }, use_nucleus_sampling=True)


def imageSplitter(lazyimage, model, vis_processors,device) :
    print("Image Splitter")
    return [ get_caption(im, model, vis_processors,device) for im in lazyimage.load()]


async def applyToSeries(series, model, vis_processors, device) : 
  print("Apply To Series")
  series.apply(lambda x:imageSplitter(x, model, vis_processors, device)).to_csv(
      f"images_captionning_results/from_{series.index[0]}_to_{series.index[-1]}.csv", index_label="index" )
    


def prepareTasksLists(serieFull, nTests, batchSize, model, vis_processors, device) : 
  print("Prepare Tasks Lists")
  n = serieFull.shape[0] if nTests == -1 else nTests
  serieWork = serieFull.iloc[:n]
  res = [
       applyToSeries(serieWork.iloc[batch*batchSize : (batch+1)*batchSize], model, vis_processors, device)
      for batch in range(n//batchSize)
  ]
  
  res += [applyToSeries(serieWork.iloc[(n//batchSize)*batchSize:], model, vis_processors, device)]
  return res


#Ensuite executer asyncio.gather sur le retour de prepareTasksLists(df['images'], 370, 37, model, vis_processors) pour tester sur 10 tasks de 37 lignes
#Si fonctionnel éxécuter asyncio.gather sur le retour de prepareTasksLists(df['images'], -1, 370, model, vis_processors)
#Je vous envoie la fonction pour parser les résultats des batchs en csv dans le dossier 'images_captionning_results' une fois les tasks terminées


#Ensuite executer asyncio.gather sur le retour de prepareTasksLists(df['images'], 370, 37, model, vis_processors) pour tester sur 10 tasks de 37 lignes
async def main():
    start = time.time()
    await asyncio.gather(*prepareTasksLists(df['images'], 100, 37, model, vis_processors,device))
    end = time.time()
    print((end - start)/60, "minutes")
    # await asyncio.gather(*prepareTasksLists(df['images'], -1, 370, model, vis_processors,device))


asyncio.run(main())


