### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 51e8bf67-4e0c-4b93-81e7-04c2b72e9b5d
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 3424d249-3fd1-4186-89a5-9fa351997ffe
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, ArchGDAL

# ╔═╡ 7d0ad20a-8cbe-11ef-192f-91f4358aeef8
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 5a271a02-e5eb-494d-9f1c-c3861f25e7c6
@bind City Select(["abudhabi", "mumbai", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "nantes", "paris", "pisa", "rennes", "saclay_e"])

# ╔═╡ 65f13a2e-8adc-4122-aedc-a59499f64a6e
folder = joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_1_rect")

# ╔═╡ a9b97b5b-084a-4d50-95fc-5f87fbeff5f4
folder_2 = joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_2_rect")

# ╔═╡ 1629ede1-f080-436d-8bc8-f2275e004e62
Images = glob("*.tif", folder)

# ╔═╡ 27985b03-2c6d-4394-857b-3c300019e205
Images_2 = glob("*.tif", folder_2)

# ╔═╡ 5a78fada-7122-4050-a776-1aa0de1db594
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ d3b76210-6d80-4059-a2f9-6ad8d1a9e150
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ e7447239-82fe-4cb8-b168-e173b51b537e
Array = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in Images]

# ╔═╡ 553e201f-b022-4985-a2f2-0058f5cd2557
data = cat(Array..., dims=3)

# ╔═╡ 06cb64ee-2821-496b-a163-b189a720ded5
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 1e96a640-63ed-4b92-82d8-6af8c8e0eae6
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, data[:, :, band])
	fig
end

# ╔═╡ be060b2b-d738-4a21-bc03-30f3f100269f
function affinity(cube; max_nz=10, chunksize=minimum(size(cube)[1:2]),
	func = c -> exp(-2*acos(clamp(c,-1,1))))

	# Verify that chunksize divides the total number of pixels
	mod(prod(size(cube)[1:2]), chunksize) == 0 ||
		error("chunksize must divide the total number of pixels")

	# Compute normalized spectra (so that inner product = cosine of angle)
	X = permutedims(reshape(cube, :, size(cube,3)))
	X = mapslices(normalize, X; dims=1)

	# Find nonzero values (in chunks)
	C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
	s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
	nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
		# Compute cosine angles (for chunk) and store in buffer
		mul!(C_buf, X', view(X, :, chunk))

		# Zero out all but `max_nz` largest values
		nzs = map(chunk, eachcol(C_buf)) do col, c
			idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
			collect(idx), fill(col, max_nz), func.(view(c,idx))
		end

		# Log progress and return
		@logprogress chunk_idx/(size(X,2) ÷ chunksize)
		return nzs
	end

	# Form and return sparse array
	rows = reduce(vcat, getindex.(nz_list, 1))
	cols = reduce(vcat, getindex.(nz_list, 2))
	vals = reduce(vcat, getindex.(nz_list, 3))
	return sparse([rows; cols],[cols; rows],[vals; vals])
end

# ╔═╡ 848fbb00-0234-44ab-961f-18b2dd07d107
max_nz = 20

# ╔═╡ 5c9cf7a1-3769-4f71-8781-9aca01d11d8b
A = cachet(joinpath(CACHEDIR, "Affinity_$City$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ edd98460-be68-42d6-a6f9-8fdc44e4e975
function embedding(A, k; seed=0)
	# Set seed for reproducibility
	Random.seed!(seed)

	# Compute node degrees and form Laplacian
	d = vec(sum(A; dims=2))
	Dsqrinv = sqrt(inv(Diagonal(d)))
	L = Symmetric(I - (Dsqrinv * A) * Dsqrinv)

	# Compute eigenvectors
	decomp, history = partialschur(L; nev=k, which=:SR)
	@info history

	return mapslices(normalize, decomp.Q; dims=2)
end

# ╔═╡ 554696b7-f9cb-4319-92f0-80ae067da104
n_clusters = 8

# ╔═╡ 6180ed8d-9cf6-495a-b0c5-b7c2aaa04583
V = embedding(A, n_clusters)

# ╔═╡ 0f81098e-778f-4bb8-9605-14e0737b3e2f
function batchkmeans(X, k, args...; nruns=100, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

# ╔═╡ 5bbd22d6-1a5c-4e4e-86bb-b79e6b78a068
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 986cff8b-278b-4291-b8b7-db7f6806ef65
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ 3c9335f8-ded3-4868-8687-b591d08fe3e3
min_index = argmin(costs)

# ╔═╡ 9e343cfb-557e-491c-bd19-943d246fb80e
clusters = spec_clusterings[min_index].assignments

# ╔═╡ f72cf08c-b6a0-4c0e-be46-86cd92740084
mat = reshape(clusters, size(data, 1), size(data, 2))

# ╔═╡ fc094227-a8c6-45f8-815a-196de403b286
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	hm = heatmap!(ax, mat; colormap=Makie.Categorical(colors))
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 8ee5f262-9f72-45ea-8d2e-5d534da4ddad


# ╔═╡ Cell order:
# ╠═7d0ad20a-8cbe-11ef-192f-91f4358aeef8
# ╠═51e8bf67-4e0c-4b93-81e7-04c2b72e9b5d
# ╠═3424d249-3fd1-4186-89a5-9fa351997ffe
# ╠═5a271a02-e5eb-494d-9f1c-c3861f25e7c6
# ╠═65f13a2e-8adc-4122-aedc-a59499f64a6e
# ╠═a9b97b5b-084a-4d50-95fc-5f87fbeff5f4
# ╠═1629ede1-f080-436d-8bc8-f2275e004e62
# ╠═27985b03-2c6d-4394-857b-3c300019e205
# ╠═5a78fada-7122-4050-a776-1aa0de1db594
# ╠═d3b76210-6d80-4059-a2f9-6ad8d1a9e150
# ╠═e7447239-82fe-4cb8-b168-e173b51b537e
# ╠═553e201f-b022-4985-a2f2-0058f5cd2557
# ╠═06cb64ee-2821-496b-a163-b189a720ded5
# ╠═1e96a640-63ed-4b92-82d8-6af8c8e0eae6
# ╠═be060b2b-d738-4a21-bc03-30f3f100269f
# ╠═848fbb00-0234-44ab-961f-18b2dd07d107
# ╠═5c9cf7a1-3769-4f71-8781-9aca01d11d8b
# ╠═edd98460-be68-42d6-a6f9-8fdc44e4e975
# ╠═554696b7-f9cb-4319-92f0-80ae067da104
# ╠═6180ed8d-9cf6-495a-b0c5-b7c2aaa04583
# ╠═0f81098e-778f-4bb8-9605-14e0737b3e2f
# ╠═5bbd22d6-1a5c-4e4e-86bb-b79e6b78a068
# ╠═986cff8b-278b-4291-b8b7-db7f6806ef65
# ╠═3c9335f8-ded3-4868-8687-b591d08fe3e3
# ╠═9e343cfb-557e-491c-bd19-943d246fb80e
# ╠═f72cf08c-b6a0-4c0e-be46-86cd92740084
# ╠═fc094227-a8c6-45f8-815a-196de403b286
# ╠═8ee5f262-9f72-45ea-8d2e-5d534da4ddad
