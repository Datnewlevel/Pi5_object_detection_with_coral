from tflite_support import metadata

populator_dst = metadata.MetadataPopulator.with_model_file('best_edgetpu.tflite')

with open('best.tflite', 'rb') as f:
  populator_dst.load_metadata_and_associated_files(f.read())

populator_dst.populate()
updated_model_buf = populator_dst.get_model_buffer()