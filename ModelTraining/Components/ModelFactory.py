from .Models.CfC import PerDeviceCfC
from .Models.FC import PerDeviceFC
from .Models.LSTM import PerDeviceLSTM
from .Models.RNN import PerDeviceRNN


class ModelFactory:
    @staticmethod
    def create_model(
        model_class: type,
        shared_input_size: int,
        per_device_input_size: int,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        n_devices: int | None = None,
    ):
        if model_class.__name__ == PerDeviceCfC.__name__:
            if n_devices is None or hidden_size is None:
                raise ValueError("n_devices and hidden_size must be provided for CfC model")
            return PerDeviceCfC(shared_input_size, per_device_input_size, n_devices, hidden_size)
        elif model_class.__name__ == PerDeviceLSTM.__name__:
            if n_devices is None or hidden_size is None or num_layers is None:
                raise ValueError("n_devices, hidden_size, and num_layers must be provided for LSTM model")
            return PerDeviceLSTM(shared_input_size, per_device_input_size, n_devices, hidden_size, num_layers)
        elif model_class.__name__ == PerDeviceRNN.__name__:
            if n_devices is None or hidden_size is None or num_layers is None:
                raise ValueError("n_devices, hidden_size, and num_layers must be provided for RNN model")
            return PerDeviceRNN(shared_input_size, per_device_input_size, n_devices, hidden_size, num_layers)
        elif model_class.__name__ == PerDeviceFC.__name__:
            if shared_input_size is None or per_device_input_size is None or n_devices is None or hidden_size is None or num_layers is None:
                raise ValueError("shared_input_size, per_device_input_size, n_devices, hidden_size, and num_layers must be provided for FC model")
            return PerDeviceFC(shared_input_size, per_device_input_size, n_devices, hidden_size, num_layers)
        else:
            raise ValueError(f"Unknown model class: {model_class.__name__}")
