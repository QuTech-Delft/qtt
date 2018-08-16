pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile.build'
            args '-v pip_cache:/var/pip_cache'
        }
    }

    stages {
        stage('Install QCoDeS')
        {
            steps {
                sh 'rm -fr Qcodes'
                sh 'git clone https://github.com/VandersypenQutech/Qcodes.git'
                sh 'virtualenv venv'
                sh '''
                       . ./venv/bin/activate &&
                       cd Qcodes && pip3 install -r requirements.txt &&
                       python3 setup.py build &&
                       python3 setup.py install &&
                       python3 -c "import qcodes"
                   '''

            }
        }

        stage('Install QTT')
        {
            steps {
                sh '''
                       . ./venv/bin/activate &&
                       pip3 install -r requirements.txt &&
                       python3 setup.py build &&
                       python3 setup.py develop  &&
                       python3 -c "import qtt"
                   '''
            }
        }

        stage('Test')
        {
            steps {
                sh '''
                       . ./venv/bin/activate &&
                       pip3 install pytest opencv-python &&
                       coverage run --source="./qtt" -m pytest -k qtt --ignore qtt/legacy.py &&
                       coverage report &&
                       coverage xml
                   '''
            }
        }

        stage('Test notebooks')
        {
            steps {
                sh '''
                       . ./venv/bin/activate &&
                       pip3 install jupyter
                   '''
                sh '. ./venv/bin/activate && jupyter --version'
                sh '. ./venv/bin/activate && jupyter nbconvert --to notebook --execute docs/notebooks/example_ohmic.ipynb'
                sh '. ./venv/bin/activate && jupyter nbconvert --to notebook --execute docs/notebooks/example_anticrossing.ipynb'
                sh '. ./venv/bin/activate && jupyter nbconvert --to notebook --execute docs/notebooks/example_PAT_fitting.ipynb'               
                sh '. ./venv/bin/activate && jupyter nbconvert --to notebook --execute docs/notebooks/example_charge_sensor.ipynb'               
                sh '. ./venv/bin/activate && jupyter nbconvert --to notebook --execute docs/notebooks/example_classical_dot_simulation.ipynb'               
            }
        }
                
        stage('Collect') {
            steps {
                step([$class: 'CoberturaPublisher', coberturaReportFile: 'coverage.xml'])
            }
        }
    }
}
