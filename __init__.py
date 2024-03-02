from utils import ImgClassifier


path = 'dataset'

MM = ImgClassifier()

dataset_generated = MM.Preprocessing(img_path=path);

model = MM.PreTrained_model(mdl=None,dense_Neuron=128,dropout=0.3,output=102)

MM.compile_and_fit_model(model,dataset_generated,epochs=100,steps_per_epoch=50)

MM.save_model(model)