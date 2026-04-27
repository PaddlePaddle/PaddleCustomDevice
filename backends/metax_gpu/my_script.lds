SECTIONS
{
  OVERLAY :
  {
    .mc_fatbin { *(.mc_fatbin) }
  }
}
INSERT AFTER .comment;
