# NMMO_competition:

Competition is from: https://www.aicrowd.com/challenges/ijcai-2022-the-neural-mmo-challenge

TODO list:<br/>
>Install all requirements<br/>
>Send and run a first submission<br/>
>Read NMMO paper<br/>
>

# Installation:

Go to your project folder, then :

      source <yourVenv>/bin/activate
      pip install git+http://gitlab.aicrowd.com/henryz/ijcai2022nmmo.git
      #Put the starter kit repo in ./ and rename it ./starter_kit/, if no starter kit install with "git clone http://gitlab.aicrowd.com/neural-mmo/ijcai2022-nmmo-starter-kit.git"
      pip install nmmo[cleanrl]
      git clone https://github.com/neuralmmo/baselines nmmo-baselines   #repo used for learning NMMO from tutorial
      echo YOUR_WANDB_API_KEY > nmmo-baselines/wandb_api_key            #optional
      git clone https://github.com/neuralmmo/environment                
      git clone https://github.com/neuralmmo/client                     

For see episode in client, run ./client/UnityClient/truc.exe (aller dans les fichiers et placer le en barre des taches c'est plus pratique pour après)
Your program need to have .render() calls or RENDER = True in config otherwise it will not render and will just run fast.

For submitting, run:

      cd starter-kit && python tool.py test                         (for a test)
      cd starter-kit && python tool.py submit <mySubmissionName>    (for a TRUE submission to AIcrowd)
