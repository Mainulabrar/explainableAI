from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
import numpy as np


# List of saved images
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0focusedSaliency{i}.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0focusedSaliency{i}fullline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0Saliency{i}Perturbationspreadfulline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0Saliency{i}Perturbationspread.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ValueSaliency{i}fulline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ValueSaliency{i}AdjustedPerturbed.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ValueSaliency{i}AdjustedPerturbedFullline.png' for i in range(18)]
# image_files = [f'/data2/mainul/ExplainableAIResultsAllSteps/figures/0ShapGraph{i}.png' for i in range(18)]
# test_set = ['014', '015', '023', '073', '098']

test_set = ['098']

for patient in test_set:
    StepNumIndicator = np.load(f'/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/{patient}tpptuning120499.npz')
    repSize = StepNumIndicator['l1'].nonzero()[0].size
    NetSet = ['0', '19999', '39999', '59999', '79999', '99999', '119999', '139999', '159999', '179999', '199999', '219999', '223999']


    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraph{i}.png' for i in range(repSize-1)]

    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesTotalTensorFinalStateBaselineGraph{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValues2ndActionZeroBase{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}SimpleGradSaliencyStep{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombined{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombined2nd{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/figures/{patient}IGValuesGraphzeroBaseCombinedSideBySide{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesLSTMGraphStep{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesGraphzeroBaseLSTMSideBySide{i}.png' for i in range(repSize-1)]
    # image_files = [f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInitdebug/figures/{patient}IGValuesGraphzeroBaseCombinedLSTMSideBySide{i}.png' for i in range(repSize-1)]
    # image_files = [f"/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/figures/{patient}IGValuesLSTMGraphCombinedStep0.png" for net in NetSet]
    # image_files = [f"/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/figures/{patient}IGValuesLSTMGraphCombinedStep1.png" for net in NetSet]
    image_files = [f"/data2/mainul/explainableAIResultsMultiplePatientCases/AllTrainingSteps/PaperInitLSTM{net}/figures/{patient}IGValuesLSTMGraphCombinedStep9.png" for net in NetSet]

    Image.MAX_IMAGE_PIXELS = None

    for file in image_files:
        with Image.open(file) as img:
        # img = Image.open(file)
            img = img.resize((1920, 1080))  # Example: Resize to 1080p resolution
            img.save(file)


    # Create a video from images
    clip = ImageSequenceClip(image_files, fps=1)  # Set frame rate
    # clip.write_videofile("animationfocusedSaliency.mp4", codec="mpeg4")
    # clip.write_videofile("animationSaliencyValueAdjustedPerturbationSpreadFullline.mp4", codec="mpeg4")
    # clip.write_videofile("animationShap.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationIG.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationTotalPolicyFinalStateIG.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationTotalPolicyTrivialZeroBaselineIG.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationTotalPolicyOptimizedFromZeroBaselineIG.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animation2ndActionZeroBaselineIG.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animation2ndActionZeroBaselineIG.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationSimpleGadinetSaliency.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationIGCombined.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationIGCombined2nd.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationIGCombinedSideBySide.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationIGLSTM.mp4", codec="mpeg4")
    # clip.write_videofile(f"{patient}animationIGLSTMSideBySide.mp4", codec="mpeg4")  
    # clip.write_videofile(f"{patient}animationIGLSTMCombinedSideBySide.mp4", codec="mpeg4")  
    # clip.write_videofile(f"{patient}animationIGLSTMfirstStep098AllNet.mp4", codec="mpeg4") 
    # clip.write_videofile(f"{patient}animationIGLSTM2ndStep098AllNet.mp4", codec="mpeg4") 
    clip.write_videofile(f"{patient}animationIGLSTM10thStep098AllNet.mp4", codec="mpeg4")       
# clip.write_videofile("animationSaliencyValueAdjustedPerturbationSpread.mp4", codec="mpeg4")
# clip.write_videofile("animationvalueSaliencyfulline.mp4", codec="mpeg4")
# clip.write_videofile("animationSaliencyPerturbationSpread.mp4", codec="mpeg4")
