import os
import sys
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import evaluate
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
import torch


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_dataloader, eval_dataloader):
        try:

            rouge_score = evaluate.load("rouge")

            logging.info("Rogue Score Defined")

            generated_summary = '''An average of two children per day were killed in Afghanistan last year, an independent Afghan rights watchdog says. The Afghanistan Rights Monitor said in a report that, of the 2,421 civilians the group registered as killed in conflict-related security incidents in 2010, some 739 were under the age of 18. It attributed almost two thirds of the child deaths to "armed opposition groups" (AOGs), or insurgents, and blamed US and NATO-led forces for 17%. The ARM report said many of the reported child casualties occurred in the violent southern provinces of Kandahar and Helmand, the traditional strongholds of the Taliban insurgency'''
            
            reference_summary = '''Nearly 740 kids—an average of about two a day—were killed in the Afghanistan war last year, says a watchdog group. About two-thirds of those deaths were blamed on insurgents—or "armed opposition groups" in the language of the report—with 17% blamed on US and NATO forces. The kids were among the record-high 2,421 Afghan civilians killed in war-related violence in 2010, reports Reuters. The number of deaths among kids is actually down from 1,050 in 2009, but the independent Afghanistan Rights Monitor says it\'s still inexcusable. "Children were highly vulnerable to the harms of war but little was done by the combatant sides, particularly by the AOGs, to ensure child safety and security during military and security incidents." All this comes against the backdrop of an escalating air war, notes Wired.'''

            scores = rouge_score.compute(predictions=[generated_summary], references=[reference_summary])

            model_checkpoint = "facebook/bart-large-cnn"

            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            optimizer = AdamW(model.parameters(), lr=2e-5)
            

            accelerator = Accelerator()
            model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader
            )

            num_train_epochs = 8
            num_update_steps_per_epoch = len(train_dataloader)
            num_training_steps = num_train_epochs * num_update_steps_per_epoch
            
            lr_scheduler = get_scheduler( "linear", optimizer=optimizer, num_warmup_steps=0,
                                         num_training_steps=num_training_steps,
            )

            def postprocess_text(preds, labels):
                 preds = [pred.strip() for pred in preds]
                 labels = [label.strip() for label in labels]
                 
                 # ROUGE expects a newline after each sentence
                 
                 preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
                 labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
                 return preds, labels
            
            model_name = "text_summarization_model"
            output_dir = "model/"

            progress_bar = tqdm(range(num_training_steps))
            
            for epoch in range(num_train_epochs):
                # Training
                model.train()
                for step, batch in enumerate(train_dataloader):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

            # Evaluation
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"], attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)
        
            # Compute metrics
            result = rouge_score.compute()
            # Extract the median ROUGE scores
            result = {key: value * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            print(f"Epoch {epoch}:", result)

            # Save and upload                
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
            
        except Exception as e:
            raise CustomException(e,sys)