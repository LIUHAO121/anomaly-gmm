import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.LSTMVAEGMM2 import LSTMVAEGMM2Primitive

class LSTMVAEGMM2SKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=LSTMVAEGMM2Primitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False