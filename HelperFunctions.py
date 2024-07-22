
import numpy as np

class HelperFunctions:
    """ 
    This class contains a series of functions that aid in computing common calculations in data science. 

    Methods: 
    condProb - calculates the conditional probability of B given A 

    medAbDev - calculates the median absolute deviation 

    normalizedError - calculates the normalized error of the input by the specified power

    catCounter - calculates the frequency of unique items 

    empiricalSampleBounds - calculates specified upper and lower bounds of a distribution 

    slidingStats - calculates the indicated descriptive statistic for the input data within the range of the specified 
    window over the entirety of the dataset 
    """

    def condProb(jointProb, probA):
      
      """
      This function calculates and returns the probability of B given A.

      @param jointProb - the joint probability of A and B
      @param probA - the probability of A
      @return answer - the conditional probability rounded to 2 decimal places
      """

      #testing invalid inputs
      if jointProb > 1 or jointProb < 0:
        print("\nPlease enter a valid probability for the first argument.")

      elif probA < 0 or probA > 1:
        print("\nPlease enter a valid probability for the second argument.")

      elif jointProb > probA:
        print("\nThe joint probability of A and B cannot be larger than the probability of A. Please try again.")

      #calculating joint probability
      else:
        answer = jointProb/probA
        
        #validating values
        if answer < 0 or answer > 1:
          print("\nA probability cannot be smaller than 0 or larger than 1. Please try again.")

        #formatting response to two decimal places
        else:
          answer = round(answer, 2)
        
        return answer

    #TODO: validate that input is indeed a numpy array     
    def medAbDev(inputArr):
      """
      This functions calculates the median absolute deviation of a given one dimensional numpy array of any size.

      @param inputArr - a one dimensional numpy array of any size containing ints/floats
      @return medianAbDev - the median absolute deviation of the provided array 
      """

      #find the median
      medianInputArr = np.median(inputArr)

      #subtract the median from every value in the array and take the abs val
      absoluteMedianDeviations = np.absolute(inputArr - medianInputArr)

      #find the median of the deviations
      medianAbDev = np.median(absoluteMedianDeviations)

      return medianAbDev
    
    def normalizedError(inputData, power):
      """
      This function calculates and outputs the normalized error of a 2d numpy array normalized by the input power.

      @param inputData - A 2D numpy array containing the predictions in column 1 and measurements in column 2
      @param power - the power by which you want to normalize the error
      @return normalizedError - the normalized error formatted to 4 decimal places
      """
      #check for appropriate shape, if not then fix it 
      if inputData.shape[0] == 2:
        inputData = inputData.transpose()
      
      #variable for summation over points
      totalSum = 0 

      #calculating the top term within the nth root of the normalization error calculation
      for index in range(len(inputData)):
        #the difference between the predicted and measured value 
        diff = np.absolute(inputData[index, 0] - inputData [index, 1])
        #raise the difference to the nth power
        nthPower = diff**power
        #add the difference to our summation variable 
        totalSum += nthPower
      
      #dividing our sum by n rows 
      insideValue = totalSum/len(inputData)
      #taking the nth root of that value 
      normalizedError = insideValue**(1/power)

      
      return round(normalizedError, 4)
    
    def catCounter(inputArr):
      """
      This function takes in a 1 dimensional array (list, frame, np array) of ints and returns
      the identified unique items and their frequency in the form of a
      2d numpy array

      @param inputArray - a 1d input 
      @return freqPairs - a 2d numpy array containing the unique values and
      their frequencies
      """
      
      #use a dictionary to keep track of items and their frequencies 
      if all(isinstance(x, int) for x in inputArr):
        freqDict = {}

        #calculating unique items and their frequencies 
        for num in inputArr:

          #if it's not in our dictionary, add it and start the value at 1
          if num not in freqDict.keys():
            freqDict[num] = 1

          #if it's in our dictionary, add 1 to the value 
          else:
            freqDict[num] += 1
      
        #turn the dictionary into a numpy array 
        freqPairs = np.array(list(freqDict.items()))

        return freqPairs
      
      else:
        print("All inputs must be integers.")
    
    def empiricalSampleBounds(inputArr, probBounds):
      """
      This function takes in a one dimensional input array and the probability mass bounds of interest.

      @param inputArr - the array containing the data in question 
      @param probBounds - the probability mass bounds that are being calculated 
      @return lowerBound, upperBound - the calculated lower and upper bound, respectively 
      """
      #sort input array for ease of use
      sortedArr = np.sort(inputArr)

      #number of points in each percentile 
      samplesInPercentile = len(sortedArr)/100

      #calculate the lower and upper percentiles of interest 
      lowPerc = (100 - probBounds)/2
      highPerc = 100 - lowPerc

      #find the idices that correspond to the values at those bounds in our sample
      lowerIndex = round(samplesInPercentile*lowPerc)-1
      upperIndex = round(samplesInPercentile*highPerc)-1

      #using our indices, find the lower and upper bound values and round to 3 decimal places
      lowerBound = round(sortedArr[lowerIndex], 3)
      upperBound = round(sortedArr[upperIndex], 3)

      return lowerBound, upperBound
    
    def slidingStats(inputData, flag, window):
      """
      This function calculates the descriptive statistics for the specified input
      data within the range of the specified window for the entirety of the dataset. 

      @param inputData - a one or two dimensional array containing integers/floats.
      Note that only flag 3 assumes a two dimensional input array. 

      @param flag - the flag with which to identify which calculation should be executed.
      1 = mean, 2 = population standard deviation, 3 = pearson correlation coefficient

      @param window - the window specifying the range of data inputs
      from which you would like the calculation 

      @return returnArr - an array containing the specified calculations from the 
      specified window
      """
      
      #Set up a np.array of the appropriate size to contain our output values 
      length = len(inputData) - (window - 1)
      returnArr = np.empty(length)
      returnArr[:] = np.NaN

      #Instantiate our index tracker variables 
      startIndex = 0
      endIndex = window

      match flag:
        
        #calc the mean over the sliding window 
        case 1:
          #until we reach the end 
          while endIndex <= len(inputData):
            #take an array containing points in our window
            tempArr = inputData[startIndex:endIndex]
            #calculate the mean of that window
            calcMean = np.mean(tempArr)
            #store it in our return array 
            returnArr[startIndex] = calcMean
            #increase our indices for the next window calculation 
            startIndex +=1
            endIndex += 1
          
          return returnArr

        #calculate the standard deviation
        case 2:
          #until we reach the end
          while endIndex <= len(inputData):
            #take an array containing points in our window 
            tempArr = inputData[startIndex:endIndex]
            #calculate the standard deviation of that window
            calcStd = np.std(tempArr)
            #store it in our return array 
            returnArr[startIndex] = calcStd
            #increase our indices for the next window calculation 
            startIndex +=1
            endIndex += 1
          
          return returnArr

        #calculate the pearson correlation coefficient
        case 3:
          #until we reach the end 
          while endIndex <= len(inputData):
            #take an array containing points in our window 
            tempArr = inputData[startIndex:endIndex]
            #calculate the pearson correlation coefficient for our window 
            calcCor = np.corrcoef(tempArr[:,0], tempArr[:,1])
            #extract the value 
            actualR = calcCor[0,1]
            #store it in our return array 
            returnArr[startIndex] = actualR
            #increase our indices for the next window 
            startIndex +=1
            endIndex += 1
          
          return returnArr

        case default:
          print('Please enter a valid flag (1, 2, 3).')
    





      






    





      



