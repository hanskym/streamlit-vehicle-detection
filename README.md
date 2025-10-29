# Streamlit App: Vehicle Detection with YOLOv8

A lightweight and responsive Streamlit web application for **Vehicle Detection**. This project utilizes a fine-tuned **YOLOv8s** model to specifically identify three primary vehicle categories: **Bus**, **Car**, and **Van** in uploaded images.

## Technology & Workflow

-   **Deployment:** This application is deployed and hosted using **Streamlit Cloud**.
-   **Core Model:** Leverages **YOLOv8s** (You Only Look Once, version 8 small), known for its high speed and accuracy in object detection.
-   **Model Training:** The model was trained and optimized using the **Google Colab** environment.
-   **Web Application:** The interactive interface is built with **Streamlit**, allowing users to run inference directly through the browser.

🔗 **Training Repository:** Full details on the model training process can be accessed here: [AI Engineering Capstone – Capstone 4](https://github.com/hanskym/ai-engineering-capstone/tree/main/capstone-4)

## Application Features

The app is designed for an intuitive user experience, with all processing running **locally** within Streamlit.

-   **Interactive User Interface:**
    -   **Image Upload:** Supports common image formats (**JPG/PNG**).
    -   **Adjustable Confidence Threshold:** Users can control the detection sensitivity.
-   **Visualization & Analysis:**
    -   **Side-by-Side View:** Displays the original image and the annotated results (_bounding boxes_) for easy visual comparison.
    -   **Automatic Vehicle Counting:** Provides a detailed summary and count for each detected vehicle class.
-   **Built-in Functionality:**
    -   **Theme Support:** Fully compatible with Streamlit's dark and light themes.
    -   **Error Handling:** Integrated error management features.
    -   **Download Results:** Allows users to download the annotated detection result image.
