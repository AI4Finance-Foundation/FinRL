import torch
import torch.nn as nn
from ..layers.Pyraformer_EncDec import Encoder


class Model(nn.Module):
    """ 
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs, window_size=[4,4], inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        if self.task_name == 'short_term_forecast':
            window_size = [2,2]
        self.encoder = Encoder(configs, window_size, inner_size)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, self.pred_len * configs.enc_in)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, configs.enc_in, bias=True)
        elif self.task_name == 'classification':
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.encoder(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.encoder(x_enc, x_mark_enc=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
