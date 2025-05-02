pipenv install --dev --skip-lock

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

docker run --rm --gpus all nvcr.io/nvidia/tensorflow:24.03-tf2-py3 \
 python - <<'EOF'
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
EOF
