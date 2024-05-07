In the initiative entitled "Road Scene Understanding for the Visually Impaired", our research team is meticulously advancing the development of the Sidewalk Environment Detection System for Assistive NavigaTION (hereinafter referred to as SENSATION). The primary objective of this venture is to enhance the mobility capabilities of blind or visually impaired persons (BVIPs) by ensuring safer and more efficient navigation on pedestrian pathways.
For the implementation phase, a specialized apparatus has been engineered: a chest-mounted bag equipped with an NVIDIA Jetson Nano, serving as the core computational unit. This device integrates a plethora of sensors including, but not limited to, tactile feedback mechanisms (vibration motors) for direction indication, optical sensors (webcam) for environmental data acquisition, wireless communication modules (Wi-Fi antenna) for internet connectivity, and geospatial positioning units (GPS sensors) for real-time location tracking.
Despite the promising preliminary design of the prototype, several technical challenges persist that warrant investigation.
The "Road Scene Understanding for the Visually Impaired" initiative is actively seeking student collaborators to refine the Jetson Nano-fueled SENSATION system. Through the combination of GPS systems and cutting-edge image segmentation techniques refined for sidewalk recognition, participating teams are expected to architect an application tailored to aid BVIPs in traversing urban landscapes, seamlessly guiding them from a designated starting point to a predetermined destination.
The developmental framework for this endeavor is based on the Python programming language; hence, prior experience with Python will be an advantage.
For the purposes of real-world testing and calibration, the navigation part will start at the main train station in Erlangen and end at the University Library of Erlangen-Nuremberg (Schuhstrasse 1a).
Technical milestones that must be completed in this project include:
Algorithmic generation of navigational pathways in Python, depending on defined start and endpoint parameters.
Real-time geospatial tracking to determine the immediate coordinates of the BVIP.
Optical recording of the current coordinates and subsequent algorithmic evaluation to check the orientation of the sidewalk


FAU Box Links

input_video.mp4 : https://faubox.rrze.uni-erlangen.de/getlink/fi4SkMw7qgsHNDEmYtSQR5/input_video.mp4
model.onnx : https://faubox.rrze.uni-erlangen.de/getlink/fiQHYEVH7FSYSk9pfskf8o/model.onnx
trained on cityscapes model checkpoint : https://faubox.rrze.uni-erlangen.de/getlink/fiQxx8EmbRenukfSUVyJpY/trained_on_cityscapes.ckpt
fine-tuned on mapillary model checkpoint :https://faubox.rrze.uni-erlangen.de/getlink/fiVwCRYbMxHR2ZnoxcNnXb/fine_tuned_mapillary.ckpt

