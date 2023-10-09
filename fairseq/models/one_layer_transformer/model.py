from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, base_architecture

@register_model('one_layer_transformer')
class CustomTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        # Add custom arguments here

    @classmethod
    def build_model(cls, args, task):
        # Ensure all settings are set to custom values before building base architecture
        args.encoder_embed_dim = 128
        args.encoder_ffn_embed_dim = 512
        args.encoder_layers = 1
        args.encoder_attention_heads = 4
        args.decoder_embed_dim = 128
        args.decoder_ffn_embed_dim = 512
        args.decoder_layers = 1
        args.decoder_attention_heads = 4
        return super().build_model(args, task)
    

@register_model_architecture('one_layer_transformer', 'one_layer_transformer_architecture')
def custom_transformer_architecture(args):
    base_architecture(args)
    # Add custom architecture changes here