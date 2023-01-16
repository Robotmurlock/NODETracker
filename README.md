# NODETracker

...

## Skupovi podataka

Pregled skupova podataka.

### MOT20

Skup scena sa fiksiranim kamerama gde se prate pesaci. Detaljan opis
se moze pronaci u [radu](https://arxiv.org/pdf/2003.09003.pdf).

Skup podataka se moze preuzeti [ovde](https://motchallenge.net/data/MOT20/).

### LaSOT

U izradi...

## Podesavanje okruzenja

Neophodno je da se instaliraju paketi navedeni u `requirements.txt`: 
`pip install -r requirements.txt`

## Skripte

Ova sekcije ukratko objasnjava neophodne korake za podesavanje
i pokretanje skripti. 
Sve glavne skripte se nalaze u `tools` direktorijumu.

### Podesavanje (konfiguracija)

Sve `tools` skripte koriste zajednicka podesavanja koja sadrze
sledece elemente:
- `resources`: podesavanje resursa (broj procesorski jezgara, da li se koristi CUDA, ...);
- `dataset`: podesavanje skupa podataka (treniranje, validacija, ...);
- `transform`: podesavanja transformacija podataka;
- `model`: podesavanje modela (izbor modela i hiperparametri);
- `train`: podesavanje za proces treniranja modela (epochs, batch_size, ...);
- `eval`: podesavanje za proces evaluacije.
- `visualize`: podesavanje za proces vizualizacije rezultata.

Prilikom pokretanja svake `tools` skripte se cuva istorija pokrenutih podesavanja.

### Alati (skripte)

Trenutni skup `tools` skripti je:
- `run_train`: treniranje modela;
- `run_inference`: predvidjanje na zadatom skupu i evaluacija.
- `run_visualize`: vizualizacija rezultata.

### Struktura generisanih podataka:

Svi generisani podaci se cuvaju na `master_path` putanji koja je 
podesiva. Struktura:

```
{dataset_name}/
    {experiment_name}/
        checkpoints/*
        configs/*
            train.yaml (sacuvane konfiguracije za pokrenute skripte)
            inference.yaml
            visualize.yaml
        inferences/*
            {inference_name}/*
                [visualize]/*   
                ...
        tensorboard_logs/*
```