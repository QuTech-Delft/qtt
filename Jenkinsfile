pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile.build'
        }
    }
    stages {
        stage('Install qcodes')
        {
            steps {
                sh 'cd Qcodes && ls -als'
                sh 'cd Qcodes && pip3 install -r requirements.txt'
                sh 'cd Qcodes && python3 setup.py build'
                sh 'cd Qcodes && python3 setup.py build'
                sh 'cd Qcodes && python3 setup.py install --user'
                sh 'python3 -c "import qcodes"'
            }
        }

        stage('Install QTT')
        {
            steps {
                sh 'pip3 install redis'
                sh 'pip3 install -r requirements.txt'
                sh 'python3 setup.py build'
                sh 'python3 setup.py develop --user'
            }
        }
    }
}
