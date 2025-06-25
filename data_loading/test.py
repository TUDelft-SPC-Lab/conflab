import sys
sys.path.append("/home/zonghuan/tudelft/projects")
# import conflab
from conflab.data_loading.pose import ConflabPoseExtractor, ConflabToKinetics
from conflab.data_loading.accel import ConflabAccelExtractor
from conflab.data_loading.person import ConflabDataset
from conflab.constants import processed_pose_path, processed_wearables_path

pose_extractor = ConflabPoseExtractor(processed_pose_path)
# accel_extractor = ConflabAccelExtractor(raw_wearables_path_new)
accel_extractor = ConflabAccelExtractor(processed_wearables_path)
pose_extractor.load_data()
examples = pose_extractor.make_examples()
ds = ConflabDataset(examples, {
    'pose': pose_extractor,
    'accel': accel_extractor
}, transform=ConflabToKinetics())

c = 9

