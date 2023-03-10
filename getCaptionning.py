import torch, sys, os
from PIL import Image
from dataloader.get import DataGetter
from threading import Thread

sys.path.append('./LAVIS/')
from lavis.models import load_model_and_preprocess

import asyncio
import time

df=DataGetter()
df=df.getData('train')
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=True, device=device
)
vis_processors.keys()
print(f"Device : {device}")
def get_caption(image, model, vis_processors, device):
    return model.generate({
          "image":  vis_processors["eval"](Image.fromarray(image)).unsqueeze(0).to(device)
          }, use_nucleus_sampling=True)



def imageSplitter(lazyimage, model, vis_processors,device) :
    return [ get_caption(im, model, vis_processors,device) for im in lazyimage.load()]


def applyToSeries(nThThread, series, model, vis_processors, device) : 
  start = time.time()
  if not os.path.exists(f"images_captionning_results/from_{series.index[0]}_to_{series.index[-1]}.csv"):
    series.apply(lambda x:imageSplitter(x, model, vis_processors, device)).to_csv(
        f"images_captionning_results/from_{series.index[0]}_to_{series.index[-1]}.csv", index_label="index" )
  print(f"Finished {nThThread}-th thread in {time.time() - start} seconds.")
    
def executeThreads(threads, batch, batchThreadSize, n, batchSize):
    start = time.time()
    print(f"Starting batch of thread [{batch-batchThreadSize}-{batch}]/{n//batchSize}.")
    for thread in threads:
        thread.start()
    print(f"Waiting for batch [{batch-batchThreadSize}-{batch}]/{n//batchSize} to finish...")
    for thread in threads:
        thread.join()
    print(f"Batch [{batch-batchThreadSize}-{batch}]/{n//batchSize} finished in {time.time() -start } seconds !")
    threads=[]

def prepareTasksLists(serieFull, nTests, batchSize, model, vis_processors, device) : 
    n = serieFull.shape[0] if nTests == -1 else nTests
    serieWork = serieFull.iloc[:n]
    batchThreadSize = 2
    threads=[]
    for batch in range(n//batchSize):
        threads += [
            Thread(target = applyToSeries, args = (batch, serieWork.iloc[batch*batchSize : (batch+1)*batchSize], model, vis_processors, device))
        ]
        if batch %batchThreadSize ==0 and batch !=0:
            executeThreads(threads, batch, batchThreadSize, n, batchSize)
    
    if len(threads) != 0 :
        executeThreads(threads, batch, batchThreadSize, n, batchSize)

    if n%batchSize != 0 :
        last_thread = Thread(target = applyToSeries, args=(n//batchSize, serieWork.iloc[(n//batchSize)*batchSize:], model, vis_processors, device))
        last_thread.start()
        last_thread.join()


def main():
    start = time.time()
    print("Starting...")
    prepareTasksLists(df['images'], -1, 50, model, vis_processors,device)
    end = time.time()
    print("Total tasks duration : ", (end - start), "seconds")

main()