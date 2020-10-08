if(F)
{
  install.packages(c("rjson","magick"))
  library(rjson)
  library(magick)
}

doit <- function()
{
  COCO <- fromJSON(file="coco.json", method = "C", unexpected.escape = "error", simplify = TRUE)
  newCOCO <- COCO
  
  for (i in 1:length(COCO[["images"]]))
  {
    print(paste("Starting image #",i))
    if(T)
    {
      fileURL <- COCO[["images"]][[i]][["file_name"]]
      fileID <- COCO[["images"]][[i]][["id"]]
      
      print(paste("Reading image  #",i))
      fileIMG <- image_read(fileURL)
      
      newCOCO[["images"]][[i]][["file_name"]] <- paste0(fileID,".jpg")
      
      newCOCO[["licenses"]] <- NULL
      newCOCO[["images"]][[i]][["license"]] <- NULL
      newCOCO[["images"]][[i]][["flickr_url"]] <- NULL
      #newCOCO[["images"]][[i]][["coco_url"]] <- NULL
      newCOCO[["images"]][[i]][["date_captured"]] <- NULL
      
      print(paste("Writing image  #",i))
      image_write(fileIMG, paste0("images/",fileID,".jpg"), format = "jpeg")
      #save.image(fileIMG,paste("images/",fileID,".jpg"))
    }
  }
  for (i in 1:length(COCO[["categories"]]))
  {
    newCOCO[["categories"]][i][["supercategory"]] <- "ant"
  }
  
  exportJSON <- toJSON(newCOCO)
  write(exportJSON, file="images/coco.json")
}
