# Dunkelwolke

This project can primary detect darkening areas from [Solar Dynamics Observatory (SDO)](https://sdo.gsfc.nasa.gov/) video files for [19.3 nm channel of AIA (Atmospheric Imaging Assembly)](https://sos.noaa.gov/catalog/datasets/sun-iron-wavelength-aia-193/) and drawing the orthodromes of the nearest $`n`$ points. The application uses the [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) class from [scikit-learn](https://github.com/scikit-learn) for claster determination. For calculating and drawing the geodesic distance between points it using the [GeoPy](https://github.com/geopy/geopy) library. For a stable application see better [SunPy](https://docs.sunpy.org/en/stable/generated/gallery/map/image_bright_regions_gallery_example.html).

This is scientifically quite useless but draws pretty [graphics](https://vimeo.com/974144794/a31bd3a871). :sun_with_face:

### GUI
![capture](https://github.com/herdav/dunkelwolke/blob/main/img/gui-3.png)
Video courtesy of NASA/SDO and the AIA, EVE, and HMI science teams.

### Cluster
![capture](https://github.com/herdav/dunkelwolke/blob/main/img/cluster.png)

### Orthodrome
![capture](https://github.com/herdav/dunkelwolke/blob/main/img/ellipse.png)
