import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.LSTMAE import LSTMAEPrimitive

class LSTMAESKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=LSTMAEPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False