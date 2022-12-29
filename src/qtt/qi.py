"""
Created on Sat Nov 12 21:36:42 2022

@author: eendebakpt
"""

import datetime

from rich import print as rprint


def qi_backend_online(backend_name: str, verbose: bool = True) -> bool:
    """ Return True if the specified QI backend is online

    Example:
        >>> qi_backend_online('Starmon-5')

    """
    from quantuminspire.api import QuantumInspireAPI
    from quantuminspire.credentials import get_token_authentication, load_account
    QI_URL = r'https://api.quantum-inspire.com'

    token = load_account()
    authentication = get_token_authentication(token)
    qi_api = QuantumInspireAPI(QI_URL, authentication)

    bb = qi_api.get_backend_types()

    if verbose:
        print(f'{datetime.datetime.now().isoformat()}: QI backends:')
        for backend in bb:
            name = backend['name']
            status = backend['status']
            if status == 'OFFLINE':
                rprint(f'  backend {name}: [yellow]{status}[/yellow] ')
            else:
                print(f'  backend {name}: {status} ')

    lst = [b for b in bb if b['name'] == backend_name]
    if lst:
        backend = lst[0]
        return not backend['status'] == 'OFFLINE'
    else:
        raise ValueError(f'backend {backend_name} is not a QI backend')


if __name__ == '__main__':

    print(qi_backend_online('Starmon-5'))
