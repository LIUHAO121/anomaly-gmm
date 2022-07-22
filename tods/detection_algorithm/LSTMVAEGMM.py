from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout , LSTM, Lambda,Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.models.base import BaseDetector

# Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
# from numba import njit
from pyod.utils.utility import argmaxn

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas
import uuid
    
    
from d3m import container, utils as d3m_utils

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase



__all__ = ('DeepLogPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):

    hidden_size = hyperparams.Hyperparameter[int](
        default=64,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="hidden state dimension"
    )
    
    
    encoder_neurons = hyperparams.List(
        default=[1, 4, 1],
        elements=hyperparams.Hyperparameter[int](1),
        description='The number of neurons per hidden layers in encoder.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    decoder_neurons = hyperparams.List(
        default=[4, 4, 4],
        elements=hyperparams.Hyperparameter[int](1),
        description='The number of neurons per hidden layers in decoder.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    
    latent_dim = hyperparams.Hyperparameter[int](
        default=2,
        description='Number of samples per gradient update.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    

    loss = hyperparams.Hyperparameter[typing.Union[str, None]](
        default='mean_squared_error',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="loss function"
    )

    optimizer = hyperparams.Hyperparameter[typing.Union[str, None]](
        default='Adam',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Optimizer"
    )

    epochs = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Epoch"
    )

    batch_size = hyperparams.Hyperparameter[int](
        default=64,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Batch size"
    )

    dropout_rate = hyperparams.Hyperparameter[float](
        default=0.2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Dropout rate"
    )

    l2_regularizer = hyperparams.Hyperparameter[float](
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="l2 regularizer"
    )

    validation_size = hyperparams.Hyperparameter[float](
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="validation size"
    )

    window_size = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="window size"
    )

    features = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Number of features in Input"
    )

    stacked_layers = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Number of LSTM layers between input layer and Final Dense Layer"
    )

    preprocessing = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether to Preprosses the data"
    )

    verbose = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="verbose"
    )
    hidden_activation = hyperparams.Enumeration[str](
        values=['relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
                'tanh', 'selu', 'elu', 'exponential'],
        default='relu',
        description='Activation function to use for hidden layers.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    
    gamma = hyperparams.Hyperparameter[float](
        default=1.0,
        description='Coefficient of beta VAE regime. Default is regular VAE.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    capacity = hyperparams.Hyperparameter[float](
        default=0.0,
        description='Maximum capacity of a loss bottle neck.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    contamination = hyperparams.Uniform(
        lower=0.,
        upper=0.5,
        default=0.1,
        description='the amount of contamination of the data set, i.e.the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class LSTMVAEGMMPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that uses DeepLog for outlier detection

    Parameters
        ----------


    """

    metadata = metadata_base.PrimitiveMetadata({
        '__author__': "DATA Lab @Texas A&M University",
        'name': "DeepLog Anomolay Detection",
        'python_path': 'd3m.primitives.tods.detection_algorithm.deeplog',
        'source': {
            'name': "DATALAB @Taxes A&M University", 
            'contact': 'mailto:khlai037@tamu.edu',
        },
        'hyperparams_to_tune': ['hidden_size','hidden_activation', 'loss', 'optimizer', 'epochs', 'batch_size',
                                'l2_regularizer', 'validation_size', 
                                'window_size', 'features', 'stacked_layers', 'preprocessing', 'verbose', 'dropout_rate','contamination'],
        'version': '0.0.1', 
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE
        ], 
        'primitive_family': metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'DeepLogPrimitive')),
        }
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,  #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        if hyperparams['loss'] == 'mean_squared_error':
            loss = keras.losses.mean_squared_error
        else:
            raise ValueError('VAE only suports mean squered error for now')
        self._clf = LstmVAEGMM(hidden_size=hyperparams['hidden_size'],
                            encoder_neurons=hyperparams['encoder_neurons'],
                            decoder_neurons=hyperparams['decoder_neurons'],
                            latent_dim=hyperparams['latent_dim'],
                            loss=loss,
                            optimizer=hyperparams['optimizer'],
                            epochs=hyperparams['epochs'],
                            batch_size=hyperparams['batch_size'],
                            dropout_rate=hyperparams['dropout_rate'],
                            l2_regularizer=hyperparams['l2_regularizer'],
                            validation_size=hyperparams['validation_size'],
                            window_size=hyperparams['window_size'],
                            stacked_layers=hyperparams['stacked_layers'],
                            preprocessing=hyperparams['preprocessing'],
                            verbose=hyperparams['verbose'],
                            contamination=hyperparams['contamination'],
                            hidden_activation = hyperparams['hidden_activation'],
                            gamma=hyperparams['gamma'],
                            capacity=hyperparams['capacity']
                                )

    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Set training data for outlier detection.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        super().set_training_data(inputs=inputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """
        return super().fit()

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.

        Returns:
            Container DataFrame
            1 marks Outliers, 0 marks normal.
        """
        return super().produce(inputs=inputs, timeout=timeout, iterations=iterations)

    def get_params(self) -> Params:
        """
        Return parameters.
        Args:
            None

        Returns:
            class Params
        """
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters for outlier detection.
        Args:
            params: class Params

        Returns:
            None
        """
        super().set_params(params=params)


class LstmVAEGMM(BaseDetector):
    """Class to Implement Deep Log LSTM based on "https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf
       Only Parameter Value anomaly detection layer has been implemented for time series data"""

    def __init__(self, hidden_size : int = 64,
                 optimizer : str ='adam',loss=mean_squared_error, preprocessing=True,
                 epochs : int =100, batch_size : int =32, dropout_rate : float =0.0,
                 l2_regularizer : float =0.1, validation_size : float =0.1, encoder_neurons=None, decoder_neurons=None,
                 latent_dim=2, hidden_activation='relu',
                 output_activation='sigmoid',gamma: float=1.0, capacity: float=0.0,
                 window_size: int = 1, stacked_layers: int  = 1, verbose : int = 1, contamination:int = 0.001):

        super(LstmVAEGMM, self).__init__(contamination=contamination)
        
        self.hidden_size = hidden_size
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.window_size = window_size
        self.stacked_layers = stacked_layers
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.dropout_rate = dropout_rate
        self.contamination = contamination
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.capacity = capacity
        

    def sampling(self, args):
        """Reparametrisation by sampling from Gaussian, N(0,I)
        To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
        with latent variables z: z = z_mean + sqrt(var) * epsilon

        Parameters
        ----------
        args : tensor
            Mean and log of variance of Q(z|X).
    
        Returns
        -------
        z : tensor
            Sampled latent variable.
        """

        z_mean, z_log = args
        batch = K.shape(z_mean)[0]  # batch size
        timesteps = K.shape(z_mean)[1] # timestemps
        dim = K.int_shape(z_mean)[2]  # latent dimension
        epsilon = K.random_normal(shape=(batch, timesteps, dim))  # mean=0, std=1.0
        
  

        return z_mean + K.exp(0.5 * z_log) * epsilon

    def vae_loss(self, inputs, outputs, z_mean, z_log,energy_out):
        """ Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """

        reconstruction_loss = self.loss(inputs, outputs)  # y_true [batch_size, d0, .. dN]. loss -> [batch_size, d0, .. dN-1]
        reconstruction_loss *= self.n_features_
        kl_loss = 1 + z_log - K.square(z_mean) - K.exp(z_log)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        kl_loss = self.gamma * K.abs(kl_loss - self.capacity)
        
        return K.mean(reconstruction_loss + kl_loss) + K.mean(energy_out)


    
    def sample_energy(self,gamma_i,z_i, mu_i, sigma_i):
        
        """
        gamma_i,:(timesteps,k)
        z_i:  (timesteps,l)
        mu_i:  (k,l)
        sigma_i: (k,l,l)
        
        """
        z_i = z_i[-1]
        gamma_i = gamma_i[-1]
        e=tf.convert_to_tensor(0)
        cov_eps = tf.eye(mu_i.shape[1]) * (1e-12)
        n_gmm = mu_i.shape[0]
        zi = zi[None,:]
        for k,gamma_i_k in enumerate(gamma_i):
            mu_i_k = mu_i[k][:, None]  # (l,1)
            d_k = zi - miu_k   # (l,1)
            inv_cov = tf.raw_ops.MatrixInverse(input=sigma_i[k] + cov_eps)
            matmul_3 = tf.matmul(tf.matmul(tf.transpose(d_k,[1,0]),inv_cov),d_k)
            e_k = tf.math.exp(-0.5 * matmul_3)
            
            e_k = e_k / tf.sqrt(tf.math.abs((tf.linalg.det(2 * 3.1415 * sigma_i[k]))))
            e_k = e_k * gamma_i_k
            e += tf.squeeze(e_k)           
        return -tf.log(e)
            
    @tf.function
    def energy(self, gamma, z):
        """
        calculate energy for every sample in z
        gamma: (batch,timesteps,k)
        z: (batch,timesteps,l)
        
        i   : index of samples
        k   : index of components
        t   : index of time
        l,m : index of features
        
        return n_samples
        """
        gamma_sum = tf.reduce_sum(gamma, axis=1) # (i,k)
        mu = tf.einsum('itk,itl->ikl',gamma,z) / gamma_sum[:,:,None]  # (i,k,l) 每个sample之间的mu和sigma都是独立的
        z_centered = tf.sqrt(gamma[:,:,:,None]) * (z[:,:,None,:] - mu[:,None, :, :]) # (i,t,k,l)
        sigma = tf.einsum('itkl,itkm->iklm',z_centered,z_centered) / gamma_sum[:,:,None,None] # (i,k,l,m) 
        
        z_centered = z[:,:,None,:] - mu[:,None, :, :] # (i,t,k,l) -> (i,k,1,l) or (i,k,l,1)
        z_centered_last = z_centered[:,-1,:,:]
        z_c_left = z_centered_last[:,:,None,:]
        z_c_right = z_centered_last[:,:,:,None]
        inverse_sigma = tf.linalg.inv(sigma)  # (i,k,l,m)
        matrix_matmul = tf.squeeze(tf.matmul(tf.matmul(z_c_left,inverse_sigma),z_c_right)) # (i,k)
        
        e_i_k = tf.math.exp(-0.5 * matrix_matmul) # (i,k)
        det_i_k = tf.sqrt(tf.math.abs((tf.linalg.det(2 * 3.1415 * sigma)))) # (i,k)
        
        e = tf.reduce_sum((e_i_k / det_i_k) * gamma[:,-1,:],axis=-1)
        
        return -tf.math.log(e)
  

    def _build_model(self):
        """
        Builds Stacked LSTM model.
        Args:
            inputs : Self object containing model parameters

        Returns:
            return : model
        """
        
        
        inputs = Input(shape=(None, self.n_features_,), name="inputs")

        # return_sequences=True ！！！！
        layer = LSTM(self.hidden_size,return_sequences=True,dropout = self.dropout_rate)(inputs)
        
         # Hidden layers
        for neurons in self.encoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation,
                          activity_regularizer=l2(self.l2_regularizer))(layer)
            layer = Dropout(self.dropout_rate)(layer)

        z_mean = Dense(self.latent_dim)(layer)
        z_log = Dense(self.latent_dim)(layer)
        # Use parametrisation sampling
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))(
            [z_mean, z_log])
        
        encoder = Model(inputs, [z_mean, z_log, z])
        if self.verbose >= 1:
            encoder.summary()

        latent_inputs = Input(shape=(None,self.latent_dim,))
        
        layer = Dense(self.latent_dim, activation=self.hidden_activation)(latent_inputs)
        
        for neurons in self.decoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation)(layer)
            layer = Dropout(self.dropout_rate)(layer)
            
        # Output layer
        outputs = Dense(self.n_features_, activation=self.output_activation)(layer)
        
        # Instatiate decoder
        decoder = Model(latent_inputs, outputs)
        if self.verbose >= 1:
            decoder.summary()
        
         # Generate outputs
        outputs = decoder(encoder(inputs)[2])
        
        
        # estimate model
        est_input = Input(shape=(None, self.n_features_,), name="est_input")
        est_outputs = Dense(self.n_features_, activation=self.output_activation)(est_input)
        est_outputs = Dense(4)(est_outputs) # (i,t,k)
        est_outputs = tf.nn.softmax(est_outputs)
        
        est_model = Model(est_input,est_outputs) 
        if self.verbose >= 1:
            est_model.summary()
            
        est_outputs = est_model(decoder(encoder(inputs)[2])) # gamma
        
        # energy calculate
        energy_input1 = Input(shape=(None, 4,), name="energy_input1")
        energy_input2 = Input(shape=(None, self.n_features_,), name="energy_input2")
        
        energy_out = self.energy(energy_input1,energy_input2)
        
        energy_model = Model([energy_input1,energy_input2],energy_out)
        if self.verbose >= 1:
            energy_model.summary()
        
        energy_out = energy_model([est_model(decoder(encoder(inputs)[2])), decoder(encoder(inputs)[2])])
        

        # lstm vae gmm
        lstmvaegmm = Model(inputs, energy_out)
        
        
        
        lstmvaegmm.add_loss(self.vae_loss(inputs, outputs, z_mean, z_log, energy_out))
        lstmvaegmm.compile(optimizer=self.optimizer)
        if self.verbose >= 1:
            lstmvaegmm.summary()
        return lstmvaegmm

    def fit(self, X, y=None):
        """
        Fit data to  LSTM model.
        Args:
            inputs : X , ndarray of size (number of sample,features)

        Returns:
            return : self object with trained model
        """


        X = check_array(X)
        self._set_n_classes(y)
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]


        X_train,Y_train = self._preprocess_data_for_LSTM(X)

        data = {
            "inputs":X_train,
            # "targets":Y_train,
        }

        self.model_ = self._build_model()
        self.history_ = self.model_.fit(data,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history
        pred_scores = np.zeros(X.shape)
        pred_scores[self.window_size-1:] = self.model_.predict(X_train)[:,-1,:] # 输出的shape 为(n,timestemps,features) 但是要用(n,features)

        Y_train_for_decision_scores = np.zeros(X.shape)
        Y_train_for_decision_scores[self.window_size-1:] = Y_train
        self.decision_scores_ = pairwise_distances_no_broadcast(Y_train_for_decision_scores,
                                                                pred_scores)

        self._process_decision_scores()
        return self


    def _preprocess_data_for_LSTM(self,X):
        """
        Preposses data and prepare sequence of data based on number of samples needed in a window
        Args:
            inputs : X , ndarray of size (number of sample,features)

        Returns:
            return : X , Y  X being samples till (t-1) of data and Y the t time data
        """
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:   # pragma: no cover
            X_norm = np.copy(X)

        X_data = []
        Y_data = []
        for index in range(X.shape[0] - self.window_size + 1):
            X_data.append(X_norm[index:index+self.window_size])
            Y_data.append(X_norm[index+self.window_size - 1])
        X_data = np.asarray(X_data)
        Y_data = np.asarray(Y_data)

        return X_data,Y_data





    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. .
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])

        X = check_array(X)
        #print("inside")
        #print(X.shape)
        #print(X[0])
        X_norm,Y_norm = self._preprocess_data_for_LSTM(X)
        pred_scores = np.zeros(X.shape)
        pred_scores[self.window_size-1:] = self.model_.predict(X_norm)[:,-1,:]
        Y_norm_for_decision_scores = np.zeros(X.shape)
        Y_norm_for_decision_scores[self.window_size-1:] = Y_norm
        return pairwise_distances_no_broadcast(Y_norm_for_decision_scores, pred_scores)
