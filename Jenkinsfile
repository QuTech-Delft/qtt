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

        stage('Install QC-Toolkit')
        {
            steps {
                sh 'rm -fr qc-toolkit'
                sh 'git clone https://github.com/qutech/qc-toolkit.git'
                sh 'cd qc-toolkit && pip3 install -r requirements.txt'
                sh 'cd qc-toolkit && python3 setup.py build'
                sh 'cd qc-toolkit && python3 setup.py install --user'
                sh 'python3 -c "import qctoolkit"'
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
                sh 'pip3 list'
                sh 'coverage run --source="./qtt" -m pytest -k qtt --ignore qtt/legacy.py'
                sh 'coverage report'
                sh 'coverage xml'
            }
        }
        
        stage('Collect') {
            steps {
                step([$class: 'CoberturaPublisher', coberturaReportFile: 'coverage.xml'])
            }
        }
    }
}
