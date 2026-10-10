I want to walk you through a project I built to solve a high-volume data problem: processing multi-terabyte spatial telemetry from NASA's Dawn mission to model the surface of the asteroid Vesta.

What drew me to this problem was a fundamental engineering bottleneck. When you take satellite imagery of an irregularly shaped body in space, every single pixel represents a unique combination of latitude, longitude, solar incidence angle, and spacecraft viewing angle. Traditional processing scripts were reading these image files frame by frame, running out of RAM, and storing intermediate results in unindexed flat files. Querying a specific surface region meant re-reading terabytes of raw data from disk every single time.

I decided to re-architect this from first principles.

First, I built a ray-tracing ingestion pipeline in Python using SPICE geometry kernels. For every image frame, it maps individual camera pixels to the facet coordinates of a 3D shape model. This turned raw 2D images into a structured, spatial time-series dataset of over 5 billion data points.

Second, to make this data actually queryable, I bypassed traditional relational databases entirely. I built a columnar storage layer using Apache Parquet partitioned by spatial regions and illumination geometry, using DuckDB and Dask to run in-memory SQL aggregations across distributed nodes on a Slurm cluster. This reduced query latency from minutes of disk-bound waiting to sub-second SQL lookups.

Third, I focused heavily on reliability and validation. Sensor data from spacecraft carries noise, missing frames, and edge-case geometries. I wrote property-based data quality checks using PyTest to catch unphysical angles and schema drifts before the data hit our analytical models. Once validated, the pipeline feeds these clean phase-curve aggregates into MCMC sampling algorithms to fit non-linear photometric models of surface roughness.

Why am I proud of it? Because it takes a messy, multi-terabyte physical dataset and turns it into a clean, reproducible, and fast data engine. It forced me to handle real scale, think critically about memory structures and disk I/O, and build automated testing so that when a model runs, I know the underlying data is 100% reliable.

Thanks you.
