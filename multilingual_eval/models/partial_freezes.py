from transformers.models.bert.modeling_bert import BertPreTrainedModel


def bitfit(model: BertPreTrainedModel, encoder_getter=None):

    if isinstance(model, BertPreTrainedModel):
        for name, param in model.bert.named_parameters():
            if not name.endswith(".bias"):
                param.requires_grad = False

    return model
