import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from pvlib.location import Location


# TODO: implement cloud_cover_to_irradiance_clearsky_scaling
def generate_pv_data(start: str, end: str, freq: str, pv_system: dict) -> np.ndarray:
    """
    Genera i dati di produzione energetica di un sistema fotovoltaico (PV) nel tempo specificato.

    Args:
        start (str): Data e ora di inizio della simulazione nel formato 'YYYY-MM-DD HH:MM'.
        end (str): Data e ora di fine della simulazione nel formato 'YYYY-MM-DD HH:MM'.
        freq (str): Frequenza temporale per i dati generati (es. 'H' per oraria).
        pv_system (dict): Dizionario contenente le informazioni sul sistema fotovoltaico,
                          con le seguenti chiavi:
                          - 'kwp': Potenza del sistema in kWp.
                          - 'provincia': Nome della provincia per la geolocalizzazione.
                          - 'opzionali' (dict, opzionale):
                              - 'altitudine': Altitudine in metri (default 0).
                              - 'tilt': Inclinazione dei pannelli in gradi (default 30).
                              - 'azimuth': Azimut dei pannelli in gradi (default 180).
                              - 'efficienza': Efficienza del pannello solare in percentuale (default 18).

    Returns:
        np.ndarray: Array contenente la potenza prodotta in kW per ciascun intervallo di tempo,
                    arrotondata a due decimali.
    """
    kwp = pv_system['kwp']

    geo_nom = Nominatim(user_agent='ceritalia')
    location = geo_nom.geocode(pv_system['provincia'])
    location_lat, location_lon = location.latitude, location.longitude

    altitude = pv_system.get('opzionali', {}).get('altitudine') or 0
    pv_location = Location(location_lat, location_lon, tz='Europe/Rome', altitude=altitude)

    times = pd.date_range(start=start, end=end, freq=freq, tz=pv_location.tz)
    clear_sky = pv_location.get_clearsky(times)
    ghi = clear_sky['ghi'].values  # global horizontal irradiance
    dni = clear_sky['dni'].values  # direct normal irradiance
    dhi = clear_sky['dhi'].values  # diffuse horizontal irradiance

    tilt = pv_system.get('opzionali', {}).get('tilt') or 30
    azimuth = pv_system.get('opzionali', {}).get('azimuth') or 180
    efficiency = pv_system.get('opzionali', {}).get('efficienza') or 18

    # Solar panel area (in square meters)
    # Convert efficiency from percentage to decimal
    panel_area = (kwp * 1000) / (efficiency * 1000 / 100)

    # Convert tilt and azimuth to radians
    tilt_rad = np.radians(tilt)
    azimuth_rad = np.radians(azimuth)

    # Calculate the angle of incidence on the tilted surface
    # Simple model assuming a fixed tilt and azimuth
    cos_zenith = np.cos(np.radians(90 - tilt))  # Simplified calculation for demo purposes

    # Calculate the irradiance on the tilted surface
    irradiance_on_panel = (dni * cos_zenith + dhi * (1 + np.cos(tilt_rad)) / 2) * 0.9  # Simplified

    # Calculate power output in watts
    power_output_watts = irradiance_on_panel * panel_area * (efficiency / 100)  # in watts

    # Convert power output to kW
    power_output_kw = power_output_watts / 1000

    return np.round(power_output_kw, 2)
