write <- function(..., sep=" ", collapse=" ") {
    cat(paste(..., sep=sep, collapse=collapse))
}


writeln <- function(..., sep=" ", collapse=" ") {
    cat(paste(paste(..., sep=sep, collapse=collapse), "\n"))
}
