#!/usr/bin/env python3
import tensorflow as tf


class FiLMGen:
  def __init__(self,
    hidden_dim=512,
    gamma_option='linear',
    gamma_baseline=1,
    num_modules=4,
    module_num_layers=1,
    module_dim=128
  ):  
    self.gamma_option = gamma_option
    self.gamma_baseline = gamma_baseline
    self.num_modules = num_modules
    self.module_num_layers = module_num_layers
    self.module_dim = module_dim
    self.cond_feat_size = 2 * self.module_dim * self.module_num_layers  # FiLM params per ResBlock


  def _build_graph(self,input_image):
    x = input_image

    pass





  def get_dims(self, x=None):
    V_in = self.encoder_embed.num_embeddings
    V_out = self.cond_feat_size
    D = self.encoder_embed.embedding_dim
    H = self.encoder_rnn.hidden_size
    H_full = self.encoder_rnn.hidden_size * self.num_dir
    L = self.encoder_rnn.num_layers * self.num_dir

    N = x.size(0) if x is not None else None
    T_in = x.size(1) if x is not None else None
    T_out = self.num_modules
    return V_in, V_out, D, H, H_full, L, N, T_in, T_out


  def encoder(self, x):
    
    return 

  def decoder(self, encoded, dims, h0=None, c0=None):
    V_in, V_out, D, H, H_full, L, N, T_in, T_out = dims

    if self.decoder_type == 'linear':
      # (N x H) x (H x T_out*V_out) -> (N x T_out*V_out) -> N x T_out x V_out
      return self.decoder_linear(encoded).view(N, T_out, V_out), (None, None)

    encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
    if not h0:
      h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))

    if self.decoder_type == 'lstm':
      if not c0:
        c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
      rnn_output, (ht, ct) = self.decoder_rnn(encoded_repeat, (h0, c0))
    elif self.decoder_type == 'gru':
      ct = None
      rnn_output, ht = self.decoder_rnn(encoded_repeat, h0)

    rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
    linear_output = self.decoder_linear(rnn_output_2d)
    if self.output_batchnorm:
      linear_output = self.output_bn(linear_output)
    output_shaped = linear_output.view(N, T_out, V_out)
    return output_shaped, (ht, ct)

  def forward(self, x):
    if self.debug_every <= -2:
      pdb.set_trace()
    encoded = self.encoder(x)
    film_pre_mod, _ = self.decoder(encoded, self.get_dims(x=x))
    film = self.modify_output(film_pre_mod, gamma_option=self.gamma_option,
                              gamma_shift=self.gamma_baseline)
    return film

  def modify_output(self, out, gamma_option='linear', gamma_scale=1, gamma_shift=0,
                    beta_option='linear', beta_scale=1, beta_shift=0):
    gamma_func = None
    beta_func = None

    gs = []
    bs = []
    for i in range(self.module_num_layers):
      gs.append(slice(i * (2 * self.module_dim), i * (2 * self.module_dim) + self.module_dim))
      bs.append(slice(i * (2 * self.module_dim) + self.module_dim, (i + 1) * (2 * self.module_dim)))

    if gamma_func is not None:
      for i in range(self.module_num_layers):
        out[:,:,gs[i]] = gamma_func(out[:,:,gs[i]])
    if gamma_scale != 1:
      for i in range(self.module_num_layers):
        out[:,:,gs[i]] = out[:,:,gs[i]] * gamma_scale
    if gamma_shift != 0:
      for i in range(self.module_num_layers):
        out[:,:,gs[i]] = out[:,:,gs[i]] + gamma_shift
    if beta_func is not None:
      for i in range(self.module_num_layers):
        out[:,:,bs[i]] = beta_func(out[:,:,bs[i]])
      out[:,:,b2] = beta_func(out[:,:,b2])
    if beta_scale != 1:
      for i in range(self.module_num_layers):
        out[:,:,bs[i]] = out[:,:,bs[i]] * beta_scale
    if beta_shift != 0:
      for i in range(self.module_num_layers):
        out[:,:,bs[i]] = out[:,:,bs[i]] + beta_shift
    return out
