from PIL import Image
import numpy as np

def merge_images_side_by_side(image_path1, image_path2, output_path):
    """
    Merges two images side by side and saves the output as a single image.

    :param image_path1: Path to the first image file.
    :param image_path2: Path to the second image file.
    :param output_path: Path where the merged image will be saved.
    """
    # Open the two images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Find the height of the tallest image
    max_height = max(image1.height, image2.height)

    # Resize images to the same height while maintaining aspect ratio
    image1 = image1.resize((int(image1.width * max_height / image1.height), max_height))
    image2 = image2.resize((int(image2.width * max_height / image2.height), max_height))

    # Create a new image with the combined width
    total_width = image1.width + image2.width
    merged_image = Image.new("RGB", (total_width, int(max_height*2.0)))

    # Paste the two images side by side
    merged_image.paste(image1, (0, (int(max_height/4.0))))
    merged_image.paste(image2, (image1.width, (int(max_height/4.0))))

    customdpi = (600, 600)
    # Save the merged image
    merged_image.save(output_path, dpi = customdpi)
    print(f"Merged image saved at {output_path}")

# Example usage
# test_set = ['015', '098']
test_set = ['014', '015', '023', '073', '098']

for patient in test_set:
    StepNumIndicator = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
    repSize = StepNumIndicator['l1'].nonzero()[0].size

    for i in range(repSize-1):

        # output_path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombinedSideBySide{i}.png'
        # image1Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombined{i}.png'
        # image2Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombined2nd{i}.png'

        # output_path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesGraphzeroBaseLSTMSideBySide{i}.png'
        output_path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesGraphzeroBaseCombinedLSTMSideBySide{i}.png'        
        # image1Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombined{i}.png'
        # image2Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombined2nd{i}.png'
        # image1Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesLSTMGraphStep{i}.png'
        # image2Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesLSTMGraph2ndStep{i}.png'
        image1Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesLSTMGraphCombinedStep{i}.png'
        image2Path = f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesLSTMGraph2ndCombinedStep{i}.png'        

        merge_images_side_by_side(image1Path, image2Path, output_path)