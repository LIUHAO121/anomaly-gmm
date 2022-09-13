import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.LSTMVAEMTDF import LSTMVAEMTDFPrimitive

class LSTMVAEMTDFSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=LSTMVAEMTDFPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False