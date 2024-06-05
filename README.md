# Content Moderation Website
## Introduction
Our website filters inappropriate content in both existing and generated content, ensuring it aligns with common sense and Islamic principles. We aim to create a safe and clean online environment.

## Data
### Video blurring
We merge multi dataset from roboflow to handle our model (yolov8), The tottal dataset 20,000 images.
[DATASET](https://universe.roboflow.com/jaishreeram/violence_maksad)

![robofloww](https://github.com/ibrahimAlawi/Deraa/assets/158778240/1fe5ec6b-79c4-45b2-b81a-7fac0523915e)
### Text to image

We used our dataset from kaggle to fine tune our model, 160,000 toxic comments.
[DATASET](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge?select=train.csv)

![images (1)](https://github.com/ibrahimAlawi/Deraa/assets/158778240/683e2653-b368-4887-81c9-e028a85fd6c1)
## How It Works
### Video Blurring Pipeline
- Enter the YouTube URL: Users input the YouTube URL.
- Process Video: The video is processed to blur inappropriate objects.
- Show Blurred Video: The video is displayed with the blurred objects.

### Example
![Screenshot_2024-05-30_192200](https://github.com/ibrahimAlawi/Deraa/assets/158778240/125fff3b-d247-46dd-b7a7-b6b1d16fa5bc)  

### Image Generation Pipeline
- Enter Text: Users input text.
- Moderate Text: The text is moderated to ensure it is appropriate.
- Generate Image: An image is generated based on the moderated text.
- Show Result: The resulting image is displayed.
![Screenshot_2024-05-30_192228](https://github.com/ibrahimAlawi/Deraa/assets/158778240/d89e918c-bcd3-4ca8-a3e1-c7f2db397149)

### Example
![image](https://github.com/ibrahimAlawi/Deraa/assets/158778240/e5aecfa2-23ef-4e41-b8d1-c2956d2342a3)



## Conclusion
To enhance safety on social media, we have developed two models specifically designed to moderate inappropriate content in images and videos. These models ensure a safer online environment by adhering to common sense and Islamic principles.
