'''class FiLMGen

FiLMGen.type(dtype) <-
FiLMGen.eval() <-

q_raw = "do you konw me?"
q_tokens = tokenize(q_raw)
predicted_program = FiLMGen(encoded.q_tokens)'''
import tensorflow as tf


film_gen:
    input: image + user intension
    input param: number of layers per CNN-VAE module, layer dimension of CNN-VAE layers, number of CNN-VAE modules
    body: image -> encoder
    return: film_param (gamma_beta) * number of CNN-VAE layers
    
film_net:
    input: image + film_param
    body: CNN-VAE
    return: latent_vector
    confirm: latent_vector -> decoder -> reconst_image

train:
    film_gen body: random init
    film_net body: pre-trained CNN-VAE model
