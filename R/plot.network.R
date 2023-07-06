library(RColorBrewer)

# net: bnstruct::BN
# positions: data.frame with columns x, y, and group.
plot.network <- function(
    net, positions, title, filename = NULL, save_path=c(".")
)
{

    if (!is.null(filename)) {
        filename <- do.call(file.path, as.list(c(save_path, filename)))
        writeln("Saving plot to", filename)

        pdf(filename, width = 9, height = 6)
    }

    variables <- bnstruct::variables(net)

    plot(
        net,
        method = "qgraph",
        title = title,
        layout = as.matrix(positions[variables, c("x", "y")]),
        groups = as.factor(positions[variables, "group"]),
        shape = "rectangle",
        vsize = 14,
        vsize2 = 5,
        label.cex = 1.2,
        color = brewer.pal(4, "Pastel2"),
        label.scale.equal = TRUE,
        legend = FALSE,
        bidirectional = FALSE,
    )

    if (!is.null(filename)) {
        dev.off()
    }
}