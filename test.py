from tflite_support import metadata
import os

assert os.path.exists('best.tflite'), "best.tflite không tồn tại!"
assert os.path.exists('best_edgetpu.tflite'), "best_edgetpu.tflite không tồn tại! Chạy edgetpu_compiler trước."

populator_dst = metadata.MetadataPopulator.with_model_file('best_edgetpu.tflite')

with open('best.tflite', 'rb') as f:
    populator_dst.load_metadata_and_associated_files(f.read())

populator_dst.populate()

updated_model_buf = populator_dst.get_model_buffer()
with open('best_edgetpu.tflite', 'wb') as f:
    f.write(updated_model_buf)

print("Metadata copy to best_edgetpu.tflite successfully!")
print(f"File size: {os.path.getsize('best_edgetpu.tflite')} bytes")
