# AlexNet


In deze opdracht gaan we AlexNet toepassen op onderstaande hypothetische kwestie.

## Kwestie
Tomatenteler 'De tomatador' is - met een totale afzet van ruim 100 miljoen kilo per jaar - een van de grootste telers van het noordelijk halfrond. De grootste bedreigingen voor deze tomatenteler zijn ziekten en aandoeningen op de tomatenplanten.

Hierom worden de tomatenplanten  wekelijks gecontroleerd op de aanwezigheid van ziekten. Zodra een ziekte wordt ontdekt op een plant, wordt deze behandeld en/of vernietigd. Waar nodig gebeurt dit ook met omliggende planten. Zowel de controle, de behandeling als de vernietiging gebeurt op dit moment handmatig.

Voor de productie van 10 kilo tomaten zijn zo'n twee planten nodig. Voor de ruim 100 miljoen kilo productie worden dus jaarlijks zo'n 20 miljoen tomatenplanten verwerkt. Ervan uitgaande dat een tomatenteler 250 planten per uur kan controleren, zijn per wekelijkse controleronde 80.000 arbeidsuren nodig, oftewel zo'n 2.000 FTE. De totale kostenpost voor wekelijkse handmatige controle is geschat op 40 miljoen euro per jaar.


## Data
Er zijn in totaal 18.159 foto's verzameld van bladeren van tomatenplanten. Deze data is te downloaden van [Canvas](https://canvas.hu.nl/courses/47169/files/folder/opdrachten/AlexNet?preview=6116015). Pak de zip uit en plaats de datamap in de root van dit project. De 18.159 foto's zijn verdeeld over 10 categorieën:


1. Bacterievlekkenziekte

![Bacterievlekkenziekte](img/TMBS.jpg "Bacterievlekkenziekte")

2. Alternaria-bladvlekkenziekte (ook wel vroege aardappelziekte)

![Alternaria-bladvlekkenziekte](img/TMEB.jpg "Alternaria-bladvlekkenziekte")

3. Gezond

![Gezond](img/TMHE.jpg "Gezond")

4. Phytophthora (ook wel late aardappelziekte)

![Phytophthora](img/TMLB.jpg "Phytophthora")

5. Bladschimmel

![Bladschimmel](img/TMLM.jpg "Bladschimmel")

6. Mozaïekvirus

![Mozaïekvirus](img/TMMV.jpg "Mozaïekvirus")

7. Septoria-bladvlekkenziekte

![Septoria-bladvlekkenziekte](img/TMSL.jpg "Septoria-bladvlekkenziekte")

8. Spintmijten

![Spintmijten](img/TMSM.jpg "Spintmijten")

9. Doelvlekkenziekte

![Doelvlekkenziekte](img/TMTS.jpg "Doelvlekkenziekte")

10. Gele bladkrulvirus

![Gele bladkrulvirus](img/TMYC.jpg "Gele bladkrulvirus")


## Doel

Het doel van deze opdracht is om de controle van aandoeningen van tomatenplanten te automatiseren. Deze automatisering dient te worden uitgevoerd door middel van een klassificatiesysteem voor verschillende tomatenplantenziekten op basis van foto's van tomatenplantenbladeren. Er is al een flinke basis gelegd, te vinden in:
- DatasetGenerator.py: functionaliteit voor het inladen van de data.
- AlexNetTrainer.py: functionaliteit voor het (trainen van het) classificatiemodel.
- GridSearch.py: functionaliteit voor het fitten van het beste model dmv grid search.
- ResultPlotter.py: functionaliteit voor het plotten van de resultaten.
- main.ipynb: Uitvoer en analyse van de oplossing.

De basis is echter incompleet. De volgende uitbreidingen aan de oplossing zijn nog vereist:
1. Implementeer de klasse AlexNet in alexnettrainer.py, om de classifiatie uit te kunnen voeren.
2. Kies in main.ipynb goede parameterwaarden om grid search over uit te voeren.
3. Implementeer verschillende plotfuncties in resltplotter.py en pas deze toe. 
   - Toon de algehele prestatie (bijvoorbeeld over de verschillende epochs).
   - Toon de prestaties per klasse (tenminste precision, recall, f1-score, confusion matrix). 
4. De gegeven dataset is niet gebalanceerd. De minder vertegenwoordigde ziekten zullen waarschijnlijk minder goed voorspeld worden. Implementeer de klasse BalancedDataset in datasetgenerator.py en pas deze toe, om de data te balanceren en de prestaties van je model te verhogen.
5. Verwerk de resultaten in een (Jupiter Notebook) rapport. Beschrijft hierin kort het onderzoek dat nodig was voor deze automatisering, inclusief resultaten, visualisaties en conclusies.

De volgende uitbreidingen zijn mogelijk voor deze opdracht:

6. (optioneel) Analyseer de data (bijvoorbeeld met Gradient-weighted Class Activation Mapping). Gebruik de resultaten om beter uit te kunnen leggen waarom sommige misclassificaties voorkomen. Bedenk zelf hoe.
7. (optioneel 2) Bedenk en implementeer een significante verbetering van de efficientie van de pipeline, onderbouw, onderzoek en documenteer de verbeteringen.
8. (optioneel 3) Bedenk en implementeer zelf nog een verbetering voor de oplossing.