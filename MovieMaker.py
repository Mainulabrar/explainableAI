from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image


# List of saved images
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0focusedSaliency{i}.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0focusedSaliency{i}fullline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0Saliency{i}Perturbationspreadfulline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0Saliency{i}Perturbationspread.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ValueSaliency{i}fulline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ValueSaliency{i}AdjustedPerturbed.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ValueSaliency{i}AdjustedPerturbedFullline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ShapGraph{i}.png' for i in range(18)]
image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0IGValuesGraph{i}.png' for i in range(18)]


for file in image_files:
    img = Image.open(file)
    img = img.resize((1920, 1080))  # Example: Resize to 1080p resolution
    img.save(file)

# Create a video from images
clip = ImageSequenceClip(image_files, fps=1)  # Set frame rate
# clip.write_videofile("animationfocusedSaliency.mp4", codec="mpeg4")
# clip.write_videofile("animationSaliencyValueAdjustedPerturbationSpreadFullline.mp4", codec="mpeg4")
# clip.write_videofile("animationShap.mp4", codec="mpeg4")
clip.write_videofile("animationIG.mp4", codec="mpeg4")
# clip.write_videofile("animationSaliencyValueAdjustedPerturbationSpread.mp4", codec="mpeg4")
# clip.write_videofile("animationvalueSaliencyfulline.mp4", codec="mpeg4")
# clip.write_videofile("animationSaliencyPerturbationSpread.mp4", codec="mpeg4")
