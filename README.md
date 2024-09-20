This project is a **skin disease identification system** that uses a **stacked machine learning model** to classify images of skin diseases. The system is designed to predict the type of skin condition from a user-uploaded image by leveraging the combined strength of **EfficientNetB0** and **InceptionV3** models through stacking, a powerful ensemble technique. The project has been structured to train both models individually, aggregate their predictions using a meta-classifier, and return the final disease classification.

### Key Features:
1. **Image Preprocessing**: The system uses `PIL` for image input and converts it into a numerical array compatible with TensorFlow for prediction, followed by softmax activation for output.

2. **Model Architecture**: The project employs a **stacked model** combining EfficientNetB0 and InceptionV3, both pre-trained on ImageNet. This allows it to handle complex image classification tasks while optimizing for performance by utilizing the strengths of both models.

3. **Training Process**: The models are trained using a dataset of labeled skin disease images with data augmentation and checkpointing to store the best performing models. After training, each model is saved in the `model/` directory for later use.

4. **Web Interface**: The application features a **Flask-based web interface** allowing users to upload images and receive classification results. The frontend includes:
   - An **upload form** for image submission.
   - A **results page** displaying the predicted skin disease.
   
5. **CSS Styling**: The interface is styled with responsive **CSS** for a clean and user-friendly experience.

6. **Use Case**: This system is designed for dermatologists or healthcare professionals looking for a quick and reliable way to classify skin diseases from images.

### Technologies Used:
- **Flask** for the backend web framework.
- **TensorFlow/Keras** for building and stacking the EfficientNetB0 and InceptionV3 models.
- **PIL** for image preprocessing.
- **ResNet50**, **EfficientNetB0**, and **InceptionV3** for model stacking.
- **HTML/CSS** for frontend design.

This project showcases advanced techniques in **deep learning**, **model stacking**, and **web integration** for practical healthcare applications.
