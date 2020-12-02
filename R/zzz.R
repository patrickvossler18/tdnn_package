.onUnload <- function (libpath) {
    library.dynam.unload("tdnn", libpath)
    invisible()
}
