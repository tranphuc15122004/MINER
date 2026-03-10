import torch
from torch import nn
from transformers import BertConfig
from utility.utils import MODEL_CLASSES
from models.fast import Fastformer

# Compat import cho BertModel (cũ / mới)
try:
    from transformers.modeling_bert import BertModel
except ImportError:
    from transformers.models.bert.modeling_bert import BertModel

ffconfig = BertConfig.from_json_file('models/ffconfig.json')


class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size, drop_rate):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.pretreained_model]
        self.config = config_class.from_pretrained(args.pretrained_model_path, output_hidden_states=True)

        if args.num_hidden_layers != -1:
            self.config.num_hidden_layers = args.num_hidden_layers

        if 'speedymind_ckpts' in args.pretrained_model_path:
            self.unicoder = model_class(config=self.config)
        else:
            self.unicoder = model_class.from_pretrained(
                args.pretrained_model_path,
                config=self.config
            )

        self.drop_layer = nn.Dropout(p=args.drop_rate)
        self.fc = nn.Linear(self.config.hidden_size, args.news_dim)

        if 'abstract' in self.args.news_attributes:
            self.text_att = AttentionPooling(args.news_dim, args.news_dim, drop_rate=args.drop_rate)
            self.sent_att = AttentionPooling(self.config.hidden_size, self.config.hidden_size, drop_rate=args.drop_rate)

    def sent_encode(self, inputs):
        batch_size, num_words = inputs.shape
        num_words = num_words // 2
        text_ids = torch.narrow(inputs, 1, 0, num_words)
        text_attmask = torch.narrow(inputs, 1, num_words, num_words)

        sent_vec = self.unicoder(text_ids, text_attmask)[0]  # B L D

        if 'abstract' in self.args.news_attributes:
            sent_vec = self.sent_att(sent_vec, text_attmask)
        else:
            sent_vec = torch.mean(sent_vec, dim=1)

        news_vec = self.fc(sent_vec)
        return news_vec

    def forward(self, inputs):
        vecs = []
        title = torch.narrow(inputs, 1, 0, self.args.num_words_title * 2)
        title_vec = self.sent_encode(title)
        vecs.append(title_vec)

        if 'abstract' in self.args.news_attributes:
            abs = torch.narrow(inputs, 1, self.args.num_words_title * 2, self.args.num_words_abstract * 2)
            abs_vec = self.sent_encode(abs)
            vecs.append(abs_vec)

        if len(vecs) == 1:
            return vecs[0]
        else:
            vecs = torch.cat(vecs, dim=-1).view(-1, len(vecs), self.args.news_dim)
            final_news_vector = self.text_att(vecs)
            return final_news_vector


class UserEncoder(nn.Module):
    def __init__(self, args, text_encoder=None):
        super(UserEncoder, self).__init__()
        self.args = args
        self.news_pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.news_attn_pool = AttentionPooling(args.news_dim, args.news_dim, drop_rate=args.drop_rate)

        # Dùng Fastformer với d_model = news_dim (256 cho checkpoint cũ)
        self.encoder = Fastformer(
            d_model=args.news_dim,
            num_layers=getattr(ffconfig, "num_hidden_layers", 2),
            num_heads=getattr(ffconfig, "num_attention_heads", 8),
            dropout=getattr(ffconfig, "hidden_dropout_prob", args.drop_rate),
        )

    def get_user_log_vec(
            self,
            sent_vecs,
            log_mask,
            log_length,
            attn_pool,
            pad_doc,
            use_mask=True
    ):
        user_log_vecs = self.encoder(sent_vecs, log_mask)
        return user_log_vecs

    def forward(self, user_news_vecs, log_mask, user_log_mask=False):
        user_vec = self.get_user_log_vec(
            user_news_vecs, log_mask,
            self.args.user_log_length,
            self.news_attn_pool, self.news_pad_doc,
            user_log_mask
        )
        return user_vec


class MLNR(nn.Module):
    def __init__(self, args):
        super(MLNR, self).__init__()
        self.args = args
        self.news_encoder = TextEncoder(args)
        self.user_encoder = UserEncoder(args, self.news_encoder if self.args.title_share_encoder else None)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,
                news_vecs,
                hist_sequence,
                hist_sequence_mask,
                candidate_inx,
                labels,
                compute_loss=True
                ):
        reshape_candidate = candidate_inx.view(-1,)
        candidate_vec = news_vecs.index_select(0, reshape_candidate)
        candidate_vec = candidate_vec.view(candidate_inx.size(0), candidate_inx.size(1), -1)

        reshape_hist = hist_sequence.view(-1,)
        log_vec = news_vecs.index_select(0, reshape_hist)
        log_vec = log_vec.view(hist_sequence.size(0), hist_sequence.size(1), -1)

        user_vec = self.user_encoder(
            log_vec, hist_sequence_mask, True
        ).unsqueeze(-1)

        score = torch.bmm(candidate_vec, user_vec).squeeze(-1)
        if compute_loss:
            loss = self.loss_fn(score, labels)
            return loss, score
        else:
            return score

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')['model_state_dict']
        for i in param_dict:
            key = i.replace('module.', '')
            if key not in self.state_dict().keys():
                continue
            if param_dict[i].size() != self.state_dict()[key].size():
                continue
            self.state_dict()[key].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
