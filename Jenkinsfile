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
                sh '. ./venv/bin/activate && cd Qcodes && pip3 install -r requirements.txt'
                sh '. ./venv/bin/activate && cd Qcodes && python3 setup.py build'
                sh '. ./venv/bin/activate && cd Qcodes && python3 setup.py install --user'
                sh '. ./venv/bin/activate && python3 -c "import qcodes"'
            }
        }

        stage('Install QTT')
        {
            steps {
                sh '. ./venv/bin/activate && pip3 install -r requirements.txt'
                sh '. ./venv/bin/activate && python3 setup.py build'
                sh '. ./venv/bin/activate && python3 setup.py develop --user'
                sh '. ./venv/bin/activate && python3 -c "import qtt"'
            }
        }

        stage('Test')
        {
            steps {
                sh '. ./venv/bin/activate && pip3 list'
                sh '. ./venv/bin/activate && coverage run --source="./qtt" -m pytest -k qtt --ignore qtt/legacy.py'
                sh '. ./venv/bin/activate && coverage report'
                sh '. ./venv/bin/activate && coverage xml'
            }
        }
        
        stage('Collect') {
            steps {
                step([$class: 'CoberturaPublisher', coberturaReportFile: 'coverage.xml'])
            }
        }
    }
}
