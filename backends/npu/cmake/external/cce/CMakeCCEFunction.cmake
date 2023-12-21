function(product_dir str newstr)
  if("x${str}" STREQUAL "xascend910")
    set(${newstr}
        "Ascend910A"
        PARENT_SCOPE)
  elseif("x${str}" STREQUAL "xascend310p")
    set(${newstr}
        "Ascend310P1"
        PARENT_SCOPE)
  else()
    string(SUBSTRING ${str} 0 1 _headlower)
    string(SUBSTRING ${str} 1 -1 _leftstr)
    string(TOUPPER ${_headlower} _headupper)
    set(${newstr}
        "${_headupper}${_leftstr}"
        PARENT_SCOPE)
  endif()
endfunction()
