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

# ╔═╡ aee13d96-3951-4d27-8558-304264cbb260
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 6409ca5c-ba46-47e1-be8c-b0a729493983
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, ArchGDAL

# ╔═╡ 5afddd44-8f0f-11ef-0e66-0f1b667bc230
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 7897afad-fd3f-4ee6-9c1b-b28f6e509ecb
@bind City Select(["abudhabi", "mumbai", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "nantes", "paris", "pisa", "rennes", "saclay_e"])

# ╔═╡ 0fc5bb5c-8bf0-4b4a-bc06-29b82845806a
Path = Dict("_T1" => joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_1_rect"), "_T2" => joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_2_rect"))

# ╔═╡ 3f49c041-8ea6-441d-b180-055f6edcfbfe
@bind Selected_Time Select(["_T1", "_T2"])

# ╔═╡ 887cc3a2-eb0f-489c-9407-56d7ee2c32f9
folder_path = Path[Selected_Time]

# ╔═╡ f6430874-a34b-409f-901d-5af237daf803
Images = glob("*.tif", folder_path)

# ╔═╡ 28f60569-bae2-4db7-ad77-3dd09a522456
CACHEDIR = joinpath(@__DIR__, "cache_files", "Onera")

# ╔═╡ 689f8f69-08ba-404f-a379-74d62616add3
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 959b5bde-1eb5-4216-a7d0-b8d09bcc4e10
Array = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in Images]

# ╔═╡ 66e50bf5-d161-4f56-be3f-7dd179ee3597
data = cat(Array..., dims=3)

# ╔═╡ 74893697-ca2c-4601-84bd-4f47b55306ee
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ b94b1aef-14d7-4a5b-9882-6173377db62c
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, data[:, :, band])
	fig
end

# ╔═╡ a4800ac6-e069-4e4c-baf0-d0b4111fad26
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

# ╔═╡ 39ee3ec6-6b76-484d-9677-a02aa95e2528
max_nz = 20

# ╔═╡ 50e12d02-2bc8-4913-a917-b54874b6a182
A = cachet(joinpath(CACHEDIR, "Affinity_$City$Selected_Time$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ 7c83722e-1cec-46f0-bb3e-efe529a3deaf
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

# ╔═╡ 7ceabfaf-2a78-470c-9aea-c891763e9478
n_clusters = 8

# ╔═╡ 57d191e2-cdeb-4c75-afbd-0cff68582063
V = embedding(A, n_clusters)

# ╔═╡ 27112e76-6600-497e-8d82-78c439f650ac
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

# ╔═╡ 79118cd9-f72a-46f6-ab08-dc3e8ce99d3c
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 2336f9ef-a72b-494b-8b56-fa5787589167
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ b110b602-9f47-4bab-abe8-0e40777f57e6
min_index = argmin(costs)

# ╔═╡ 68ae31a0-dadb-43e2-93b8-6b18f39c855e
clusters = spec_clusterings[min_index].assignments

# ╔═╡ 93df754c-bacb-48f8-87b6-dd877e328f7c
mat = reshape(clusters, size(data, 1), size(data, 2))

# ╔═╡ 6555baf8-5e80-4872-bbd6-4acc4ff711f7
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	hm = heatmap!(ax, mat; colormap=Makie.Categorical(colors))
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ cdc37739-fd88-46db-8f3c-e7728e3dbf0c


# ╔═╡ Cell order:
# ╠═5afddd44-8f0f-11ef-0e66-0f1b667bc230
# ╠═aee13d96-3951-4d27-8558-304264cbb260
# ╠═6409ca5c-ba46-47e1-be8c-b0a729493983
# ╠═7897afad-fd3f-4ee6-9c1b-b28f6e509ecb
# ╠═0fc5bb5c-8bf0-4b4a-bc06-29b82845806a
# ╠═3f49c041-8ea6-441d-b180-055f6edcfbfe
# ╠═887cc3a2-eb0f-489c-9407-56d7ee2c32f9
# ╠═f6430874-a34b-409f-901d-5af237daf803
# ╠═28f60569-bae2-4db7-ad77-3dd09a522456
# ╠═689f8f69-08ba-404f-a379-74d62616add3
# ╠═959b5bde-1eb5-4216-a7d0-b8d09bcc4e10
# ╠═66e50bf5-d161-4f56-be3f-7dd179ee3597
# ╠═74893697-ca2c-4601-84bd-4f47b55306ee
# ╠═b94b1aef-14d7-4a5b-9882-6173377db62c
# ╠═a4800ac6-e069-4e4c-baf0-d0b4111fad26
# ╠═39ee3ec6-6b76-484d-9677-a02aa95e2528
# ╠═50e12d02-2bc8-4913-a917-b54874b6a182
# ╠═7c83722e-1cec-46f0-bb3e-efe529a3deaf
# ╠═7ceabfaf-2a78-470c-9aea-c891763e9478
# ╠═57d191e2-cdeb-4c75-afbd-0cff68582063
# ╠═27112e76-6600-497e-8d82-78c439f650ac
# ╠═79118cd9-f72a-46f6-ab08-dc3e8ce99d3c
# ╠═2336f9ef-a72b-494b-8b56-fa5787589167
# ╠═b110b602-9f47-4bab-abe8-0e40777f57e6
# ╠═68ae31a0-dadb-43e2-93b8-6b18f39c855e
# ╠═93df754c-bacb-48f8-87b6-dd877e328f7c
# ╠═6555baf8-5e80-4872-bbd6-4acc4ff711f7
# ╠═cdc37739-fd88-46db-8f3c-e7728e3dbf0c
