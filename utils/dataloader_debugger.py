from data.vid.kinetics_dataloader import *
from data.vid.something_dataloader import *
from data.img.imagenet_dataloader import *
from data.nlp.pajama_dataloader import *
from data.nlp.fineweb_dataloader import *
from data.nlp.collator import *
from data.nlp.bigbench_dataloader import *
from data.nlp.gsm8k_dataloader import *
from data.nlp.ai2arc_dataloader import *
from utils import text_logger
import sys
from transformers import AutoTokenizer


def debug_dataloader(hparams, model_trainer, dataset_type='test'):
    print("Dataset:", hparams.dataset_name)
    dataset_type = 'train'
    print("Split:", dataset_type)
    dataset = None
    if hparams.dataset_name in ('kinetics400' , 'k400'):
        dataset = Kinetics400Dataset(hparams, split = dataset_type, transform = model_trainer.transform)
    elif hparams.dataset_name in ('something', 'smth'):
        dataset = SomethingDataset(hparams, split = dataset_type, transform = model_trainer.transform)
    elif hparams.dataset_name in ('imagenet' , 'imagenet1k'):
        dataset = ImageNetDataset(hparams, split = dataset_type, transform = model_trainer.transform)
    elif hparams.dataset_name == 'pajama':
        dataset = RedPajamaDataset(hparams)
    elif hparams.dataset_name == 'fineweb':
        dataset = FineWebDataset(hparams)
    elif "bigbench" in hparams.dataset_name:
        x = hparams.dataset_name
        dataset = BigBenchDataset(hparams, "train", x[x.find('_') + 1 :])
    elif hparams.dataset_name == "gsm8k":
        dataset = GSM8KDataset(hparams, dataset_type)
    elif hparams.dataset_name == 'ai2arc':
        dataset = AI2ArcDataset(hparams, split = dataset_type)
    else:
        raise ValueError("haven't implemented or added this dataset yet")
    
    collate_fn = None if not hparams.modality == "NLP" else NLP_HF_Collator(hparams)
    dataloader = DataLoader(dataset, batch_size=hparams.batch_size_per_device, num_workers=hparams.num_workers_per_device, persistent_workers=True, drop_last = True, shuffle = True, collate_fn = collate_fn)
    print("split", dataset_type)
    print("using dataset", hparams.dataset_name)
    print("dataloader len:", len(dataloader))
    for step_index, batch in enumerate(dataloader):
        # if step_index % 100 == 0: 
        print(f'Step Index: {step_index}')
            # print(type(batch), type(batch[0]), len(batch))
            # print('batch:', batch[0].shape)
            # print('Labels:', batch[1])
    print("Done going through", hparams.dataset_name)
