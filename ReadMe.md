DanceMA: Multi-Agents Framework for Open-ended Document-level Aspect-Based Sentiment Intensity Analysis with Informal Styles
-----------------------------
## Overview
In this work, we introduces \textbf{DanceMA}, a multi-agent framework for open-ended document-level ABSIA with informal styles. It comprises two components: Dance, which applies a divide-and-conquer principle to decompose the complex ABSIA task into the collaboration of expert agents, and MA, a Team Manager Agent that integrates outputs from different Dance teams to generate fine-grained and high-accuracy labels. Our findings reveal the significance of informal styles in ABSIA, as they convey stronger sentiment intensity in opinions for specific aspects.

![DaceMaFramework](./img/DanceMAFramework.png)
*Figure 1: An overview of our **DanceMA** framework, consisting of two core components:  
1. **Dance** (*Divide-and-Conquer Teamwork*) for open-ended document-level ABSIA with Informal Styles.  
2. **MA** (*Team Manager Agent*) for label annotation.

Our contributions are as follows:
- We introduces DanceMA, a multi-agent framework for open-ended document-level ABSIA with informal styles. Extensive experiments demonstrate the validity and superiority of Dance for document-level ABSIA, and the effectiveness of MA for fine-grained and high-accuracy data Annotation

- We release InformalABSIA, a document-level ABSIA dataset that focuses on informal styles and is annotated with our DanceMA framework. The dataset includes 1,500 long context documents from 3 domains, averaging 7.34 ACOSI tuples per document, which can contribute to the document-level ABSIA analysis.

- We highlight the importance of informal styles in ABSIA, showing their ability to convey stronger sentiment intensity to specific aspects with opinion terms with informal expressions.

![DanceCase](./img/DanceCase.png)
*Figure 2: A running case of Dance for document-level ABSIA with expert agents collaboration, more technical details are provided in Section \ref{sec:DanceMA}


### Code for **DanceMA** 
```
python ./code/Dance.py 
python ./code/MA.py
python ./code/LLM-as-Judge.py
```







