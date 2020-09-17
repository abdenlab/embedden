

class Model:

    def build(self, input_shape):
        pass

    def call(self, inputs, training=None, mask=None):
        pass

    def compile(
        self,
        optimizer='rmsprop',
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        **kwargs
    ):
        pass

    @property
    def metrics(self):
        pass

    def train_step(self, data):
        pass

    def make_train_function(self):
        pass

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    ):
        pass

    def test_step(self, data):
        pass

    def make_test_function(self):
        pass

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False
    ):
        pass

    def predict_step(self, data):
        pass

    def make_predict_function(self):
        pass

    def predict(
        self,
        x,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    ):
        pass

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        reset_metrics=True,
        return_dict=False
    ):
        pass

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        reset_metrics=True,
        return_dict=False
    ):
        pass

    def predict_on_batch(self, x):
        pass

