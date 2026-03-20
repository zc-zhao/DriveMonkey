import transformers
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if isinstance(outputs, dict):   # and 'pred_2d_loss' in outputs and 'pred_3d_loss' in outputs
            loss_dict = {}
            log_vars_dict = {}
            for name,value in outputs.items():
                if 'loss' in name and name != 'loss':
                    loss_value = value.item()
                    if loss_value == 0 and hasattr(self,'history_loss_dict'):
                        loss_value = self.history_loss_dict[name]
                    loss_dict[name] = loss_value
                if 'log_vars' in name and outputs['log_vars']:
                    for log_key, log_value in outputs['log_vars'].items():
                        log_vars_dict[log_key] = log_value
                    loss_dict.update(log_vars_dict)
                    # self.log(outputs['log_vars'])
            # self.update_history_loss_dict(outputs)
            if not hasattr(self,'history_loss_dict'):
                self.history_loss_dict = {}
            for name, value in outputs.items():
                if 'loss' in name and name != 'loss':
                    if name not in self.history_loss_dict:
                        self.history_loss_dict[name] = value.item()
                    else:
                        if value != 0:
                            self.history_loss_dict[name] = value.item()
            self.log(loss_dict)
    return (loss, outputs) if return_outputs else loss

def replace_train_compute_loss():
    transformers.Trainer.compute_loss = compute_loss
    print('Replace compute loss!!')