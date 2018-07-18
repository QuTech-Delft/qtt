pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile.build'
        }
    }

    stages {
        stage('Install QCoDeS')
        {
            steps {
                sh 'rm -fr Qcodes'
                sh 'git clone https://github.com/VandersypenQutech/Qcodes.git'
                sh 'cd Qcodes && pip3 install -r requirements.txt'
                sh 'cd Qcodes && python3 setup.py build'
                sh 'cd Qcodes && python3 setup.py install --user'
                sh 'python3 -c "import qcodes"'
            }
        }

        stage('Install QTT')
        {
            steps {
                sh 'pip3 install -r requirements.txt'
                sh 'python3 setup.py build'
                sh 'python3 setup.py develop --user'
                sh 'python3 -c "import qtt"'
            }
        }

        stage('Test')
        {
            steps {
                sh 'py.test-3 -k qtt --ignore qtt/legacy.py'
            }
        }
    }
}
